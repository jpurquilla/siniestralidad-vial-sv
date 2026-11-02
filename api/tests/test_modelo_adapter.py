"""Tests del adaptador del modelo de frecuencia (NB07).

Estos son los tests más importantes del proyecto: verifican que el servicio
se niegue a arrancar si falta o cambia log_tasa_base, y reproducen
exactamente la regresión que motiva esa validación (~415,125x de
sobreestimación silenciosa, sin que XGBoost lance ningún error).
"""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pytest
import xgboost as xgb

from app.config import settings
from app.domain.models import FeaturesCelda
from app.infrastructure.modelo.xgboost_adapter import (
    COLUMNAS_MODELO,
    ModeloFrecuenciaError,
    XGBoostFrecuenciaAdapter,
)

RUTA_MODELO = settings.ruta_modelo_frecuencia
RUTA_ESPECIFICACION = settings.ruta_especificacion_nb07

# Celda de referencia: San Salvador, 2024-06-15, franja Tarde (12-17).
# matriz_frecuencia_v2.csv: pct_urbano=0.9965, log_exposicion_v2=12.890956,
# esa celda registró 4 siniestros (train).
_FEATURES_CONOCIDAS = {
    "es_lluviosa": 1,
    "es_finde": 1,
    "dia_semana": 5,
    "periodo_agostino": 0,
    "es_feriado": 0,
    "prcp_mensual": 30.0,
    "pct_urbano": 0.9965,
    "poblacion_baja": 0,
}
_LOG_EXPOSICION_V2_CONOCIDA = 12.890956
_LOG_TASA_BASE_REAL = -12.936335060885538


@pytest.fixture
def especificacion_original() -> dict:
    return json.loads(RUTA_ESPECIFICACION.read_text(encoding="utf-8"))


def _escribir_especificacion_tamperada(tmp_path, especificacion: dict, mutar) -> "Path":
    espec = json.loads(json.dumps(especificacion))  # copia profunda
    mutar(espec["modelos"]["frecuencia_xgboost_offset_v2.pkl"])
    ruta = tmp_path / "nb07_especificacion_tamperada.json"
    ruta.write_text(json.dumps(espec), encoding="utf-8")
    return ruta


class TestArranqueExitoso:
    def test_carga_y_valida_con_artefactos_reales(self):
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, RUTA_ESPECIFICACION)
        adapter.cargar_y_validar()
        assert adapter.log_tasa_base == pytest.approx(_LOG_TASA_BASE_REAL, abs=1e-9)

    def test_prediccion_sobre_celda_conocida_en_rango_esperado(self):
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, RUTA_ESPECIFICACION)
        adapter.cargar_y_validar()

        celda = FeaturesCelda(
            **_FEATURES_CONOCIDAS, log_exposicion_v2=_LOG_EXPOSICION_V2_CONOCIDA
        )
        (lam,) = adapter.predecir_lote([celda])

        # La celda registró 4 siniestros en un solo día; con offset correcto
        # la lambda de Poisson debe ser un valor pequeño y positivo, no un
        # conteo puntual ni un valor disparado.
        assert 0.01 <= lam <= 20.0


class TestArranqueFallido:
    def test_falla_si_log_tasa_base_no_coincide(self, tmp_path, especificacion_original):
        ruta_tamperada = _escribir_especificacion_tamperada(
            tmp_path,
            especificacion_original,
            mutar=lambda info: info.__setitem__("log_tasa_base", -12.0),
        )
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, ruta_tamperada)
        with pytest.raises(ModeloFrecuenciaError, match="log_tasa_base"):
            adapter.cargar_y_validar()

    def test_falla_si_log_tasa_base_falta(self, tmp_path, especificacion_original):
        def quitar(info):
            del info["log_tasa_base"]

        ruta_tamperada = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, mutar=quitar
        )
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, ruta_tamperada)
        with pytest.raises(ModeloFrecuenciaError):
            adapter.cargar_y_validar()

    def test_falla_si_cambia_el_orden_de_columnas(self, tmp_path, especificacion_original):
        def barajar(info):
            info["columnas"] = list(reversed(info["columnas"]))

        ruta_tamperada = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, mutar=barajar
        )
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, ruta_tamperada)
        with pytest.raises(ModeloFrecuenciaError, match="columnas"):
            adapter.cargar_y_validar()

    def test_falla_si_el_modelo_no_tiene_8_features(self, tmp_path, monkeypatch, especificacion_original):
        ruta_espec = tmp_path / "nb07_especificacion.json"
        ruta_espec.write_text(json.dumps(especificacion_original), encoding="utf-8")

        class _BoosterFalso:
            def num_features(self):
                return 9

        monkeypatch.setattr(
            "app.infrastructure.modelo.xgboost_adapter.joblib.load",
            lambda ruta: _BoosterFalso(),
        )
        monkeypatch.setattr(
            "app.infrastructure.modelo.xgboost_adapter.isinstance",
            lambda obj, cls: True,
            raising=False,
        )
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, ruta_espec)
        with pytest.raises(ModeloFrecuenciaError, match="features"):
            adapter.cargar_y_validar()

    def test_falla_si_falta_el_archivo_del_modelo(self, tmp_path, especificacion_original):
        ruta_espec = tmp_path / "nb07_especificacion.json"
        ruta_espec.write_text(json.dumps(especificacion_original), encoding="utf-8")
        adapter = XGBoostFrecuenciaAdapter(tmp_path / "no_existe.pkl", ruta_espec)
        with pytest.raises(ModeloFrecuenciaError, match="No se encontró el modelo"):
            adapter.cargar_y_validar()


class TestRegresionOffset:
    """La prueba de regresión más importante del proyecto (ver README §
    'El fallo silencioso de log_tasa_base'): cargar el modelo SIN sumar
    log_tasa_base al base_margin y verificar que la predicción se dispara
    varios órdenes de magnitud, sin que XGBoost lance ningún error."""

    def test_sin_log_tasa_base_la_prediccion_se_dispara_ordenes_de_magnitud(self):
        booster = joblib.load(RUTA_MODELO)
        columnas = COLUMNAS_MODELO
        fila = [_FEATURES_CONOCIDAS[c] for c in columnas]

        dm_correcto = xgb.DMatrix(
            np.array([fila], dtype=float),
            base_margin=np.array([_LOG_EXPOSICION_V2_CONOCIDA + _LOG_TASA_BASE_REAL]),
            feature_names=columnas,
        )
        dm_sin_offset = xgb.DMatrix(
            np.array([fila], dtype=float),
            base_margin=np.array([_LOG_EXPOSICION_V2_CONOCIDA]),  # falta log_tasa_base
            feature_names=columnas,
        )

        (pred_correcta,) = booster.predict(dm_correcto)
        (pred_sin_offset,) = booster.predict(dm_sin_offset)

        ratio = pred_sin_offset / pred_correcta
        ratio_esperado = math.exp(-_LOG_TASA_BASE_REAL)  # ~415,125

        assert ratio_esperado == pytest.approx(415125.04, rel=1e-3)
        assert ratio == pytest.approx(ratio_esperado, rel=1e-2)
        assert pred_sin_offset > 1000 * pred_correcta  # de sobra para no arrancar

    def test_el_adaptador_no_permite_omitir_log_tasa_base(self):
        """FeaturesCelda ya no tiene un campo 'base_margin' que un llamador
        pueda construir a mano sin log_tasa_base: solo existe
        log_exposicion_v2, y XGBoostFrecuenciaAdapter.predecir_lote() es el
        único lugar que le suma log_tasa_base (ver xgboost_adapter.py). Este
        test prueba que pasar por la API pública del adaptador reproduce la
        predicción "correcta", no la inflada."""
        adapter = XGBoostFrecuenciaAdapter(RUTA_MODELO, RUTA_ESPECIFICACION)
        adapter.cargar_y_validar()

        celda = FeaturesCelda(**_FEATURES_CONOCIDAS, log_exposicion_v2=_LOG_EXPOSICION_V2_CONOCIDA)
        (pred_via_adapter,) = adapter.predecir_lote([celda])

        booster = joblib.load(RUTA_MODELO)
        fila = [_FEATURES_CONOCIDAS[c] for c in COLUMNAS_MODELO]
        dm_correcto = xgb.DMatrix(
            np.array([fila], dtype=float),
            base_margin=np.array([_LOG_EXPOSICION_V2_CONOCIDA + _LOG_TASA_BASE_REAL]),
            feature_names=COLUMNAS_MODELO,
        )
        (pred_esperada,) = booster.predict(dm_correcto)

        assert pred_via_adapter == pytest.approx(pred_esperada, rel=1e-6)
