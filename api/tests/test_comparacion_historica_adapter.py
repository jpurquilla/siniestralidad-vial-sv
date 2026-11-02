"""Tests del adaptador de comparación histórica (observado real vs. predicho
por el modelo de NB07, agregados por distrito sobre matriz_frecuencia_v2.csv).

Regla D1 de NB08 (exigir_particion_de_diagnostico): el conjunto de prueba es
la única fuente válida para reportar una métrica de desempeño — nunca se
mezcla con train, porque el modelo ya vio esas filas al entrenar y eso infla
la calibración/Spearman de forma artificial. matriz_frecuencia_v2.csv trae
una columna `split` (train=2022-2024, test=2025-2026); este adaptador debe
filtrar a `split == "test"` antes de agregar, igual que NB08 exige en los
notebooks.

Dos niveles: la lógica de agregación se prueba con un predictor falso (rápido,
sin tocar el modelo real); un test de integración aparte corre el modelo REAL
sobre el histórico completo y verifica que el total predicho quede en el
mismo orden de magnitud que el observado — la misma clase de regresión que
tests/test_modelo_adapter.py::TestRegresionOffset, aplicada acá: si alguien
rompe el offset, acá predicho saldría ~415,000× más grande."""

from __future__ import annotations

import csv

import pytest

from app.config import settings
from app.domain.models import FeaturesCelda
from app.infrastructure.analitica.comparacion_historica_adapter import (
    ComparacionHistoricaAdapter,
)
from app.infrastructure.modelo.xgboost_adapter import XGBoostFrecuenciaAdapter


class PredictorFalsoUnoPorFila:
    """Devuelve 1.0 por fila, sin importar el contenido — para probar que la
    suma por distrito agrega bien, no la corrección numérica del modelo."""

    def predecir_lote(self, filas: list[FeaturesCelda]) -> list[float]:
        return [1.0] * len(filas)


@pytest.fixture(scope="module")
def total_filas_test_por_distrito() -> dict[str, int]:
    conteo: dict[str, int] = {}
    with settings.ruta_matriz_frecuencia.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila["split"] == "test":
                conteo[fila["distrito_key"]] = conteo.get(fila["distrito_key"], 0) + 1
    return conteo


class TestAgregacionConPredictorFalso:
    def test_observado_coincide_con_suma_de_n_siniestros_solo_en_test(self):
        adapter = ComparacionHistoricaAdapter(settings.ruta_matriz_frecuencia, PredictorFalsoUnoPorFila())
        adapter.cargar_y_validar()
        resultado = {c.distrito_codigo: c for c in adapter.por_distrito()}

        observado_esperado_san_salvador = 0.0
        with settings.ruta_matriz_frecuencia.open(encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                if fila["distrito_key"] == "san salvador" and fila["split"] == "test":
                    observado_esperado_san_salvador += float(fila["n_siniestros"])

        assert resultado["san salvador"].observado == pytest.approx(
            observado_esperado_san_salvador, rel=1e-9
        )

    def test_predicho_es_una_fila_por_conteo_con_predictor_falso(self, total_filas_test_por_distrito):
        adapter = ComparacionHistoricaAdapter(settings.ruta_matriz_frecuencia, PredictorFalsoUnoPorFila())
        adapter.cargar_y_validar()
        resultado = {c.distrito_codigo: c for c in adapter.por_distrito()}

        assert resultado["san salvador"].predicho == pytest.approx(
            total_filas_test_por_distrito["san salvador"], rel=1e-9
        )

    def test_hay_103_distritos(self):
        adapter = ComparacionHistoricaAdapter(settings.ruta_matriz_frecuencia, PredictorFalsoUnoPorFila())
        adapter.cargar_y_validar()
        assert len(adapter.por_distrito()) == 103

    def test_nunca_agrega_filas_de_train(self, total_filas_test_por_distrito):
        """Regresión directa de la regla D1 (NB08): si alguien vuelve a leer
        el CSV completo sin filtrar por split, el total de filas agregadas
        se dispara a 676,504 (train 451,552 + test 224,952) en vez de
        quedarse en las 224,952 de la partición de prueba."""
        adapter = ComparacionHistoricaAdapter(settings.ruta_matriz_frecuencia, PredictorFalsoUnoPorFila())
        adapter.cargar_y_validar()

        total_test = sum(total_filas_test_por_distrito.values())
        predicho_total = sum(c.predicho for c in adapter.por_distrito())
        assert predicho_total == pytest.approx(total_test, rel=1e-9)
        assert predicho_total < 676_504


class TestIntegracionConModeloReal:
    """Regresión del offset, aplicada a la agregación completa: si alguien
    rompe log_tasa_base, el total predicho se dispara ~415,000× — este test
    lo detectaría porque la razón predicho/observado se saldría del rango
    [0.1, 10] con enorme margen."""

    def test_predicho_total_queda_en_el_mismo_orden_de_magnitud_que_observado(self):
        xgboost_adapter = XGBoostFrecuenciaAdapter(
            settings.ruta_modelo_frecuencia, settings.ruta_especificacion_nb07
        )
        xgboost_adapter.cargar_y_validar()

        adapter = ComparacionHistoricaAdapter(settings.ruta_matriz_frecuencia, xgboost_adapter)
        adapter.cargar_y_validar()

        resultado = adapter.por_distrito()
        total_observado = sum(c.observado for c in resultado)
        total_predicho = sum(c.predicho for c in resultado)

        razon = total_predicho / total_observado
        assert 0.1 < razon < 10, (
            f"Razón predicho/observado={razon:.4f} fuera de rango sano — "
            "revisar si se está sumando log_tasa_base al base_margin."
        )
