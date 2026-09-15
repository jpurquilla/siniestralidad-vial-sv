"""Tests del adaptador de exposición (log_exposicion_v2 = NB03 base + NB05 CAGR)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.config import settings
from app.infrastructure.modelo.exposicion_adapter import (
    ExposicionAdapter,
    ExposicionRepositoryError,
)


@pytest.fixture
def adapter() -> ExposicionAdapter:
    a = ExposicionAdapter(settings.ruta_matriz_frecuencia, settings.ruta_especificacion_nb05)
    a.cargar_y_validar()
    return a


class TestValoresConocidos:
    """Valores tomados directamente de matriz_frecuencia_v2.csv."""

    def test_san_salvador_2024_06_15(self, adapter):
        valor = adapter.log_exposicion_v2("san salvador", date(2024, 6, 15))
        assert valor == pytest.approx(12.890956, abs=1e-4)

    def test_acajutla_origen_factor_uno(self, adapter):
        valor = adapter.log_exposicion_v2("acajutla", date(2022, 1, 1))
        assert valor == pytest.approx(10.920654761778046, abs=1e-6)

    def test_acajutla_2026_06_30(self, adapter):
        valor = adapter.log_exposicion_v2("acajutla", date(2026, 6, 30))
        assert valor == pytest.approx(11.254123, abs=1e-4)


class TestExtrapolacionFutura:
    def test_extrapola_mas_alla_del_ultimo_mes_de_la_matriz(self, adapter):
        # La matriz termina en 2026-06-30; 6 meses después el factor de
        # crecimiento debe seguir el mismo CAGR, sin lanzar error.
        valor_futuro = adapter.log_exposicion_v2("acajutla", date(2026, 12, 1))
        valor_conocido = adapter.log_exposicion_v2("acajutla", date(2026, 6, 30))
        assert valor_futuro > valor_conocido


class TestArranqueFallido:
    def test_falla_si_falta_g_mensual(self, tmp_path):
        espec = json.loads(settings.ruta_especificacion_nb05.read_text(encoding="utf-8"))
        del espec["correccion_tendencia"]["g_mensual"]
        ruta = tmp_path / "nb05_tamperada.json"
        ruta.write_text(json.dumps(espec), encoding="utf-8")

        a = ExposicionAdapter(settings.ruta_matriz_frecuencia, ruta)
        with pytest.raises(ExposicionRepositoryError):
            a.cargar_y_validar()

    def test_falla_con_distrito_desconocido(self, adapter):
        with pytest.raises(ExposicionRepositoryError):
            adapter.log_exposicion_v2("distrito_inexistente", date(2024, 1, 1))
