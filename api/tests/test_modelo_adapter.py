"""Tests del adaptador del modelo de frecuencia (modelo final de NB12).

Estos son los tests más importantes del proyecto: verifican que el servicio
se niegue a arrancar si falta o cambia log_tasa_base o el orden de columnas,
y reproducen exactamente la regresión que motiva esa validación (~415,125x de
sobreestimación silenciosa, sin que XGBoost lance ningún error).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from app.config import settings
from app.domain.models import COLUMNAS_MODELO, FeaturesCelda, Franja
from app.infrastructure.modelo.xgboost_adapter import (
    CANARIA_FEATURES,
    ModeloFrecuenciaError,
    XGBoostFrecuenciaAdapter,
)

DIRECTORIO_MODELOS = settings.directorio_modelos
RUTA_ESPECIFICACION = settings.ruta_especificacion_nb12
_LOG_TASA_BASE_REAL = -12.936335060885538


@pytest.fixture(scope="module")
def especificacion_original() -> dict:
    return json.loads(RUTA_ESPECIFICACION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def adaptador_real() -> XGBoostFrecuenciaAdapter:
    adaptador = XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, RUTA_ESPECIFICACION)
    adaptador.cargar_y_validar()
    return adaptador


def _escribir_especificacion_tamperada(tmp_path: Path, especificacion: dict, mutar) -> Path:
    espec = json.loads(json.dumps(especificacion))  # copia profunda
    mutar(espec)
    ruta = tmp_path / "nb12_especificacion_tamperada.json"
    ruta.write_text(json.dumps(espec), encoding="utf-8")
    return ruta


def _celda_canaria(**cambios) -> FeaturesCelda:
    campos = {**CANARIA_FEATURES.__dict__, **cambios}
    return FeaturesCelda(**campos)


class TestArranqueExitoso:
    def test_carga_y_valida_con_artefactos_reales(self, adaptador_real):
        assert adaptador_real.log_tasa_base == pytest.approx(_LOG_TASA_BASE_REAL, abs=1e-9)

    def test_version_modelo_es_el_archivo_de_la_especificacion(self, adaptador_real, especificacion_original):
        assert especificacion_original["modelo_final"] == "ampliado"
        assert adaptador_real.version_modelo == especificacion_original["modelo"]
        assert adaptador_real.version_modelo == "frecuencia_xgboost_offset_v3.pkl"
        assert adaptador_real.fecha_entrenamiento == especificacion_original["generado"]

    def test_columnas_del_dominio_son_las_de_la_especificacion(self, especificacion_original):
        assert list(COLUMNAS_MODELO) == especificacion_original["columnas"]
        assert len(COLUMNAS_MODELO) == 12

    def test_orden_de_franjas_del_dominio_es_el_de_la_especificacion(self, especificacion_original):
        assert {f.value: f.codigo for f in Franja} == especificacion_original["codificacion_franja"]

    def test_prediccion_sobre_celda_conocida_en_rango_esperado(self, adaptador_real):
        (lam,) = adaptador_real.predecir_lote([CANARIA_FEATURES])
        assert 0.01 <= lam <= 20.0

    def test_constantes_de_la_canaria_coinciden_con_la_matriz_v3(self):
        """La canaria documenta sus valores como leídos de la matriz v3: se
        verifica aquí que sigan coincidiendo con la fila real."""
        with gzip.open(settings.ruta_matriz_frecuencia, "rt", encoding="utf-8") as fh:
            df = pd.read_csv(fh)
        fila = df[
            (df["distrito_key"] == "san salvador")
            & (df["fecha"] == "2024-06-15")
            & (df["franja_horaria"] == Franja.TARDE.value)
        ]
        assert len(fila) == 1
        fila = fila.iloc[0]
        for columna in COLUMNAS_MODELO:
            assert getattr(CANARIA_FEATURES, columna) == pytest.approx(float(fila[columna]), abs=1e-6), columna
        assert CANARIA_FEATURES.log_exposicion_v2 == pytest.approx(float(fila["log_exposicion_v2"]), abs=1e-6)

    def test_las_cuatro_franjas_dan_predicciones_distintas_y_madrugada_es_la_menor(self, adaptador_real):
        celdas = [_celda_canaria(franja_cod=f.codigo) for f in Franja]
        predicciones = dict(zip(Franja, adaptador_real.predecir_lote(celdas), strict=True))

        assert len(set(predicciones.values())) == 4
        assert min(predicciones, key=predicciones.get) == Franja.MADRUGADA


class TestArranqueFallido:
    def test_falla_si_log_tasa_base_no_coincide(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.__setitem__("log_tasa_base", -12.0)
        )
        with pytest.raises(ModeloFrecuenciaError, match="log_tasa_base"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_log_tasa_base_no_es_finito(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.__setitem__("log_tasa_base", "nan")
        )
        with pytest.raises(ModeloFrecuenciaError, match="log_tasa_base"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_log_tasa_base_falta(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.pop("log_tasa_base")
        )
        with pytest.raises(ModeloFrecuenciaError):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_cambia_el_orden_de_columnas(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.__setitem__("columnas", list(reversed(e["columnas"])))
        )
        with pytest.raises(ModeloFrecuenciaError, match="columnas"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_faltan_columnas(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.__setitem__("columnas", e["columnas"][:8])
        )
        with pytest.raises(ModeloFrecuenciaError, match="columnas"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_la_codificacion_de_franja_no_coincide(self, tmp_path, especificacion_original):
        def mutar(e):
            e["codificacion_franja"] = {"Madrugada (00-05)": 1, "Manana (06-11)": 0, "Tarde (12-17)": 2, "Noche (18-23)": 3}

        ruta = _escribir_especificacion_tamperada(tmp_path, especificacion_original, mutar)
        with pytest.raises(ModeloFrecuenciaError, match="codificacion_franja"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_modelo_final_no_se_reconoce(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.__setitem__("modelo_final", "otro")
        )
        with pytest.raises(ModeloFrecuenciaError, match="modelo_final"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_el_modelo_no_tiene_12_features(self, tmp_path, monkeypatch, especificacion_original):
        ruta_espec = tmp_path / "nb12_especificacion.json"
        ruta_espec.write_text(json.dumps(especificacion_original), encoding="utf-8")
        booster_real = joblib.load(DIRECTORIO_MODELOS / especificacion_original["modelo"])

        class _BoosterConOtrasFeatures(xgb.Booster):
            def num_features(self):
                return 9

        falso = _BoosterConOtrasFeatures(model_file=None)
        falso.load_config(booster_real.save_config())
        monkeypatch.setattr(
            "app.infrastructure.modelo.xgboost_adapter.joblib.load", lambda ruta: falso
        )
        with pytest.raises(ModeloFrecuenciaError, match="features"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta_espec).cargar_y_validar()

    def test_falla_si_falta_el_archivo_del_modelo(self, tmp_path, especificacion_original):
        ruta = _escribir_especificacion_tamperada(
            tmp_path, especificacion_original, lambda e: e.__setitem__("modelo", "no_existe.pkl")
        )
        with pytest.raises(ModeloFrecuenciaError, match="No se encontró el modelo"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, ruta).cargar_y_validar()

    def test_falla_si_falta_la_especificacion(self, tmp_path):
        with pytest.raises(ModeloFrecuenciaError, match="especificación"):
            XGBoostFrecuenciaAdapter(DIRECTORIO_MODELOS, tmp_path / "no_existe.json").cargar_y_validar()


class TestRegresionOffset:
    """La prueba de regresión más importante del proyecto (ver README §
    'El fallo silencioso de log_tasa_base'): cargar el modelo SIN sumar
    log_tasa_base al base_margin y verificar que la predicción se dispara
    varios órdenes de magnitud, sin que XGBoost lance ningún error."""

    @pytest.fixture
    def booster(self, especificacion_original):
        return joblib.load(DIRECTORIO_MODELOS / especificacion_original["modelo"])

    def _matriz(self) -> np.ndarray:
        return np.array([CANARIA_FEATURES.como_lista()], dtype=float)

    def test_sin_log_tasa_base_la_prediccion_se_dispara_ordenes_de_magnitud(self, booster):
        columnas = list(COLUMNAS_MODELO)
        dm_correcto = xgb.DMatrix(
            self._matriz(),
            base_margin=np.array([CANARIA_FEATURES.log_exposicion_v2 + _LOG_TASA_BASE_REAL]),
            feature_names=columnas,
        )
        dm_sin_offset = xgb.DMatrix(
            self._matriz(),
            base_margin=np.array([CANARIA_FEATURES.log_exposicion_v2]),  # falta log_tasa_base
            feature_names=columnas,
        )

        (pred_correcta,) = booster.predict(dm_correcto)
        (pred_sin_offset,) = booster.predict(dm_sin_offset)

        ratio = pred_sin_offset / pred_correcta
        ratio_esperado = math.exp(-_LOG_TASA_BASE_REAL)  # ~415,125

        assert ratio_esperado == pytest.approx(415125.04, rel=1e-3)
        assert ratio == pytest.approx(ratio_esperado, rel=1e-2)
        assert pred_sin_offset > 1000 * pred_correcta  # de sobra para no arrancar

    def test_el_adaptador_no_permite_omitir_log_tasa_base(self, adaptador_real, booster):
        """FeaturesCelda no tiene un campo 'base_margin' que un llamador
        pueda construir a mano sin log_tasa_base: solo existe
        log_exposicion_v2, y XGBoostFrecuenciaAdapter.predecir_lote() es el
        único lugar que le suma log_tasa_base. Pasar por la API pública del
        adaptador reproduce la predicción "correcta", no la inflada."""
        (pred_via_adapter,) = adaptador_real.predecir_lote([CANARIA_FEATURES])

        dm_correcto = xgb.DMatrix(
            self._matriz(),
            base_margin=np.array([CANARIA_FEATURES.log_exposicion_v2 + _LOG_TASA_BASE_REAL]),
            feature_names=list(COLUMNAS_MODELO),
        )
        (pred_esperada,) = booster.predict(dm_correcto)

        assert pred_via_adapter == pytest.approx(pred_esperada, rel=1e-6)
