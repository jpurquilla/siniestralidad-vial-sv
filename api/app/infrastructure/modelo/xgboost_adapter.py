"""Adaptador del modelo de frecuencia: XGBoost Booster con offset (modelo
final de NB12).

Se carga y valida UNA sola vez, en el lifespan de la aplicación (ver
app/main.py) — nunca por request. Si algo no cuadra, `cargar_y_validar()`
lanza `ModeloFrecuenciaError` y la aplicación debe negarse a arrancar: un
servicio degradado que sirve predicciones ~415,000 veces más grandes sin
lanzar ningún error es peor que no arrancar.

Qué modelo se sirve lo decide nb12_especificacion.json (campo `modelo_final`):
de ahí salen el archivo, las columnas y log_tasa_base. Este adaptador es el
ÚNICO lugar donde se suma log_tasa_base al base_margin.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from app.domain.models import COLUMNAS_MODELO, FeaturesCelda, Franja

logger = logging.getLogger(__name__)

# Valor de referencia SOLO para detectar en el arranque una
# nb12_especificacion.json corrupta, reemplazada o desactualizada. El valor
# operacional SIEMPRE se lee del JSON (ver cargar_y_validar) — esta
# constante nunca se usa para calcular una predicción, solo para comparar.
_LOG_TASA_BASE_ESPERADO = -12.936335060885538
_TOLERANCIA_LOG_TASA_BASE = 1e-9

# Valores posibles de `modelo_final` y el bloque de la especificación que
# describe cada uno. "ampliado" vive en la raíz del JSON; la variante
# base + franja, en `variante_base_franja`.
_MODELO_FINAL_AMPLIADO = "ampliado"
_MODELOS_FINALES_BASE_FRANJA = ("base + franja", "base_franja", "base+franja")

# Celda canaria: San Salvador, 2024-06-15, franja Tarde (12-17), partición
# train de data/processed/matriz_frecuencia_v3.csv.gz (esa celda registró 4
# siniestros). Origen de cada valor, leído de esa fila de la matriz v3:
#   prcp_mensual=723.8, pct_urbano=0.9965, franja_cod=2,
#   crec_parque_12m=0.043012 (único por distrito y mes),
#   pct_hogares_auto=0.50626 y pct_trabaja_otro_lugar=0.129056 (censales,
#   constantes por distrito), log_exposicion_v2=12.890956.
# tests/test_modelo_adapter.py verifica estas constantes contra la matriz.
# No se exige igualdad exacta -un conteo Poisson de una sola celda es
# ruidoso- sino que la lambda predicha caiga en un orden de magnitud
# razonable. Sin log_tasa_base la misma celda se dispara ~415,000 veces.
CANARIA_FEATURES = FeaturesCelda(
    es_lluviosa=1,
    es_finde=1,
    dia_semana=5,
    periodo_agostino=0,
    es_feriado=0,
    prcp_mensual=723.8,
    pct_urbano=0.9965,
    poblacion_baja=0,
    franja_cod=2,
    crec_parque_12m=0.043012,
    pct_hogares_auto=0.50626,
    pct_trabaja_otro_lugar=0.129056,
    log_exposicion_v2=12.890956,
)
_CANARIA_RANGO_ESPERADO = (0.01, 20.0)


class ModeloFrecuenciaError(RuntimeError):
    """El modelo o su especificación no pasaron las validaciones de arranque."""


def _leer_bloque_modelo(especificacion: dict) -> tuple[str, dict]:
    """Devuelve (modelo_final, bloque) según el campo `modelo_final`."""
    modelo_final = especificacion["modelo_final"]
    if modelo_final == _MODELO_FINAL_AMPLIADO:
        return modelo_final, especificacion
    if modelo_final in _MODELOS_FINALES_BASE_FRANJA:
        return modelo_final, especificacion["variante_base_franja"]
    raise ModeloFrecuenciaError(
        f"modelo_final={modelo_final!r} no reconocido (se esperaba "
        f"{_MODELO_FINAL_AMPLIADO!r} o {_MODELOS_FINALES_BASE_FRANJA})."
    )


def _validar_columnas_contra_dominio(columnas: list[str]) -> None:
    """Las columnas de la especificación deben coincidir, en nombre y orden,
    con las que FeaturesCelda.como_lista() entrega al modelo."""
    esperado = list(COLUMNAS_MODELO)
    if columnas != esperado:
        raise ModeloFrecuenciaError(
            "El orden de columnas de la especificación no coincide con el "
            f"normativo.\n  especificación: {columnas}\n  esperado:       {esperado}"
        )
    # Sonda: valores distintos por campo; como_lista() debe devolverlos en el
    # orden de `columnas` (detecta un como_lista() desalineado con los nombres).
    valores = {c: float(i + 1) for i, c in enumerate(columnas)}
    sonda = FeaturesCelda(**{c: valores[c] for c in columnas}, log_exposicion_v2=0.0)
    if sonda.como_lista() != [valores[c] for c in columnas]:
        raise ModeloFrecuenciaError(
            "FeaturesCelda.como_lista() no entrega las columnas en el orden de la especificación."
        )


def _validar_codificacion_franja(codificacion: dict) -> None:
    esperado = {franja.value: franja.codigo for franja in Franja}
    if codificacion != esperado:
        raise ModeloFrecuenciaError(
            "codificacion_franja de la especificación no coincide con el "
            f"dominio.\n  especificación: {codificacion}\n  dominio:        {esperado}"
        )


class XGBoostFrecuenciaAdapter:
    """Implementa PredictorFrecuenciaPort sobre el Booster del modelo final (NB12)."""

    def __init__(self, directorio_modelos: Path, ruta_especificacion: Path) -> None:
        self._directorio_modelos = directorio_modelos
        self._ruta_especificacion = ruta_especificacion
        self._booster: xgb.Booster | None = None
        self.log_tasa_base: float | None = None
        self.version_modelo: str | None = None
        self.fecha_entrenamiento: str | None = None

    def cargar_y_validar(self) -> None:
        """Carga el .pkl y la especificación, valida ambos, y corre la
        predicción canaria. Lanza ModeloFrecuenciaError si algo falla."""
        if not self._ruta_especificacion.exists():
            raise ModeloFrecuenciaError(
                f"No se encontró la especificación en {self._ruta_especificacion}"
            )

        try:
            especificacion = json.loads(self._ruta_especificacion.read_text(encoding="utf-8"))
            _, bloque = _leer_bloque_modelo(especificacion)
            nombre_modelo = str(bloque["modelo"])
            columnas = bloque["columnas"]
            log_tasa_base = float(bloque["log_tasa_base"])
            codificacion_franja = especificacion["codificacion_franja"]
            fecha_entrenamiento = str(especificacion["generado"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ModeloFrecuenciaError(
                f"{self._ruta_especificacion} no tiene la forma esperada: {exc!r}"
            ) from exc

        ruta_modelo = self._directorio_modelos / nombre_modelo
        if not ruta_modelo.exists():
            raise ModeloFrecuenciaError(f"No se encontró el modelo en {ruta_modelo}")

        booster = joblib.load(ruta_modelo)
        if not isinstance(booster, xgb.Booster):
            raise ModeloFrecuenciaError(
                f"{ruta_modelo} no contiene un xgboost.Booster (tipo: {type(booster)})"
            )

        _validar_columnas_contra_dominio(columnas)
        _validar_codificacion_franja(codificacion_franja)

        if not math.isfinite(log_tasa_base) or abs(
            log_tasa_base - _LOG_TASA_BASE_ESPERADO
        ) > _TOLERANCIA_LOG_TASA_BASE:
            raise ModeloFrecuenciaError(
                f"log_tasa_base leído de {self._ruta_especificacion.name} "
                f"({log_tasa_base}) no coincide con el valor esperado "
                f"({_LOG_TASA_BASE_ESPERADO}). Sin este valor exacto las "
                "predicciones se desvían varios órdenes de magnitud sin que "
                "XGBoost lance ningún error. Rechazando el arranque."
            )

        n_features = booster.num_features()
        if n_features != len(columnas):
            raise ModeloFrecuenciaError(
                f"El modelo tiene {n_features} features; se esperaban "
                f"{len(columnas)} ({columnas})"
            )

        # A partir de aquí el adaptador queda operativo: predecir_lote() ya
        # puede usarse para la predicción canaria.
        self._booster = booster
        self.log_tasa_base = log_tasa_base
        self.version_modelo = nombre_modelo
        self.fecha_entrenamiento = fecha_entrenamiento

        lam = self.predecir_lote([CANARIA_FEATURES])[0]
        lo, hi = _CANARIA_RANGO_ESPERADO
        if not (lo <= lam <= hi):
            raise ModeloFrecuenciaError(
                f"Predicción canaria fuera de rango: lambda={lam:.4f}, "
                f"esperado en [{lo}, {hi}]. El offset probablemente está mal "
                "aplicado (¿falta sumar log_tasa_base al base_margin?)."
            )

        logger.info(
            "Modelo de frecuencia validado: version=%s features=%d "
            "log_tasa_base=%.15f prediccion_canaria=%.4f rango_esperado=%s",
            self.version_modelo,
            n_features,
            log_tasa_base,
            lam,
            _CANARIA_RANGO_ESPERADO,
        )

    def predecir_lote(self, filas: list[FeaturesCelda]) -> list[float]:
        if self._booster is None or self.log_tasa_base is None:
            raise ModeloFrecuenciaError(
                "El adaptador no fue validado (llamar cargar_y_validar) antes de predecir."
            )
        if not filas:
            return []
        matriz = np.array([f.como_lista() for f in filas], dtype=float)
        # base_margin real de XGBoost = log_exposicion_v2 + log_tasa_base.
        # Se suma AQUÍ, en un único lugar, para que ningún llamador pueda
        # olvidarlo (es exactamente el fallo silencioso que este adaptador
        # existe para prevenir — ver ModeloFrecuenciaError más arriba).
        margenes = np.fromiter(
            (f.log_exposicion_v2 for f in filas), dtype=float, count=len(filas)
        ) + self.log_tasa_base
        dmatrix = xgb.DMatrix(matriz, base_margin=margenes, feature_names=list(COLUMNAS_MODELO))
        return self._booster.predict(dmatrix).astype(float).tolist()
