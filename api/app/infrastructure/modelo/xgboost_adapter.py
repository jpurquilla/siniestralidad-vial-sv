"""Adaptador del modelo de frecuencia: XGBoost Booster con offset (NB07).

Se carga y valida UNA sola vez, en el lifespan de la aplicación (ver
app/main.py) — nunca por request. Si algo no cuadra, `cargar_y_validar()`
lanza `ModeloFrecuenciaError` y la aplicación debe negarse a arrancar: un
servicio degradado que sirve predicciones ~415,000 veces más grandes sin
lanzar ningún error es peor que no arrancar.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from app.domain.models import FeaturesCelda

logger = logging.getLogger(__name__)

# Nombre del artefacto dentro de nb07_especificacion.json (NB07 §6.4).
_NOMBRE_MODELO = "frecuencia_xgboost_offset_v2.pkl"

# Orden normativo de columnas — nb07_especificacion.json > modelos >
# frecuencia_xgboost_offset_v2.pkl > columnas. Debe coincidir con
# FeaturesCelda.como_lista().
COLUMNAS_MODELO = [
    "es_lluviosa",
    "es_finde",
    "dia_semana",
    "periodo_agostino",
    "es_feriado",
    "prcp_mensual",
    "pct_urbano",
    "poblacion_baja",
]

_N_FEATURES_ESPERADO = 8

# Valor de referencia SOLO para detectar en el arranque un
# nb07_especificacion.json corrupto, reemplazado o desactualizado. El valor
# operacional SIEMPRE se lee del JSON (ver cargar_y_validar) — esta
# constante nunca se usa para calcular una predicción, solo para comparar.
_LOG_TASA_BASE_ESPERADO = -12.936335060885538
_TOLERANCIA_LOG_TASA_BASE = 1e-9

# Celda canaria: distrito San Salvador, 2024-06-15, franja Tarde (12-17)
# (train; matriz_frecuencia_v2.csv). Esa celda registró 4 siniestros. No se
# exige igualdad exacta -un conteo Poisson de una sola celda es ruidoso- sino
# que la lambda predicha caiga en un orden de magnitud razonable. Sin
# log_tasa_base la misma celda predice ~771,834 (ver tests/test_modelo_adapter.py).
_CANARIA_FEATURES_SIN_MARGEN = {
    "es_lluviosa": 1,
    "es_finde": 1,
    "dia_semana": 5,
    "periodo_agostino": 0,
    "es_feriado": 0,
    "prcp_mensual": 30.0,
    "pct_urbano": 0.9965,
    "poblacion_baja": 0,
}
_CANARIA_LOG_EXPOSICION_V2 = 12.890956
_CANARIA_RANGO_ESPERADO = (0.01, 20.0)


class ModeloFrecuenciaError(RuntimeError):
    """El modelo o su especificación no pasaron las validaciones de arranque."""


def _construir_features_canaria() -> FeaturesCelda:
    return FeaturesCelda(
        **_CANARIA_FEATURES_SIN_MARGEN,
        log_exposicion_v2=_CANARIA_LOG_EXPOSICION_V2,
    )


class XGBoostFrecuenciaAdapter:
    """Implementa PredictorFrecuenciaPort sobre el Booster de NB07."""

    def __init__(self, ruta_modelo: Path, ruta_especificacion: Path) -> None:
        self._ruta_modelo = ruta_modelo
        self._ruta_especificacion = ruta_especificacion
        self._booster: xgb.Booster | None = None
        self.log_tasa_base: float | None = None
        self.version_modelo = _NOMBRE_MODELO

    def cargar_y_validar(self) -> None:
        """Carga el .pkl y la especificación, valida ambos, y corre la
        predicción canaria. Lanza ModeloFrecuenciaError si algo falla."""
        if not self._ruta_modelo.exists():
            raise ModeloFrecuenciaError(f"No se encontró el modelo en {self._ruta_modelo}")
        if not self._ruta_especificacion.exists():
            raise ModeloFrecuenciaError(
                f"No se encontró la especificación en {self._ruta_especificacion}"
            )

        booster = joblib.load(self._ruta_modelo)
        if not isinstance(booster, xgb.Booster):
            raise ModeloFrecuenciaError(
                f"{self._ruta_modelo} no contiene un xgboost.Booster (tipo: {type(booster)})"
            )

        try:
            especificacion = json.loads(self._ruta_especificacion.read_text(encoding="utf-8"))
            info = especificacion["modelos"][_NOMBRE_MODELO]
            columnas = info["columnas"]
            log_tasa_base = float(info["log_tasa_base"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ModeloFrecuenciaError(
                f"{self._ruta_especificacion} no tiene la forma esperada para "
                f"{_NOMBRE_MODELO}: {exc}"
            ) from exc

        if columnas != COLUMNAS_MODELO:
            raise ModeloFrecuenciaError(
                "El orden de columnas de la especificación no coincide con el "
                f"normativo.\n  especificación: {columnas}\n  esperado:       {COLUMNAS_MODELO}"
            )

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
        if n_features != _N_FEATURES_ESPERADO:
            raise ModeloFrecuenciaError(
                f"El modelo tiene {n_features} features; se esperaban "
                f"{_N_FEATURES_ESPERADO} ({COLUMNAS_MODELO})"
            )

        # A partir de aquí el adaptador queda operativo: predecir_lote() ya
        # puede usarse para la predicción canaria.
        self._booster = booster
        self.log_tasa_base = log_tasa_base

        celda_canaria = _construir_features_canaria()
        lam = self.predecir_lote([celda_canaria])[0]
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
        margenes = np.array(
            [f.log_exposicion_v2 + self.log_tasa_base for f in filas], dtype=float
        )
        dmatrix = xgb.DMatrix(matriz, base_margin=margenes, feature_names=COLUMNAS_MODELO)
        return [float(x) for x in self._booster.predict(dmatrix)]
