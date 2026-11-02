"""Tests de las reglas de calendario (domain, pura) y del adaptador de
feriados (infrastructure, lee feriados_elsalvador_2022_2026.xlsx)."""

from __future__ import annotations

from datetime import date

import pytest

from app.config import settings
from app.domain.calendario import dia_semana, es_finde, es_lluviosa, periodo_agostino
from app.infrastructure.modelo.calendario_adapter import CalendarioAdapter


class TestReglasPuras:
    def test_es_lluviosa_mayo_a_octubre(self):
        assert es_lluviosa(date(2024, 5, 1)) == 1
        assert es_lluviosa(date(2024, 10, 31)) == 1
        assert es_lluviosa(date(2024, 4, 30)) == 0
        assert es_lluviosa(date(2024, 11, 1)) == 0

    def test_dia_semana_lunes_es_cero(self):
        assert dia_semana(date(2024, 1, 1)) == 0  # lunes 2024-01-01

    def test_es_finde_viernes_y_sabado(self):
        assert es_finde(date(2024, 1, 5)) == 1  # viernes
        assert es_finde(date(2024, 1, 6)) == 1  # sabado
        assert es_finde(date(2024, 1, 7)) == 0  # domingo
        assert es_finde(date(2024, 1, 1)) == 0  # lunes

    def test_periodo_agostino_3_a_6(self):
        assert periodo_agostino(date(2024, 8, 3)) == 1
        assert periodo_agostino(date(2024, 8, 6)) == 1
        assert periodo_agostino(date(2024, 8, 2)) == 0
        assert periodo_agostino(date(2024, 8, 7)) == 0


@pytest.fixture
def adapter() -> CalendarioAdapter:
    a = CalendarioAdapter(settings.ruta_feriados)
    a.cargar_y_validar()
    return a


class TestFeriadoDentroDeCobertura:
    def test_ano_nuevo_es_nacional(self, adapter):
        assert adapter.es_feriado(date(2022, 1, 1), "acajutla") is True
        assert adapter.es_feriado(date(2022, 1, 1), "san salvador") is True

    def test_dia_ordinario_no_es_feriado(self, adapter):
        assert adapter.es_feriado(date(2024, 3, 12), "san salvador") is False

    def test_fiestas_agostinas_solo_afectan_san_salvador(self, adapter):
        # Local (San Salvador): 3, 4 y 5 de agosto (NB03 §5.2)
        assert adapter.es_feriado(date(2024, 8, 3), "san salvador") is True
        assert adapter.es_feriado(date(2024, 8, 3), "acajutla") is False


class TestFeriadoFueraDeCobertura:
    def test_recicla_el_patron_de_fecha_fija(self, adapter):
        # Año Nuevo (1 de enero) es un feriado de fecha fija: debe reciclarse
        # igual en 2030, fuera de la cobertura 2022-2026 del archivo.
        assert adapter.es_feriado(date(2030, 1, 1), "acajutla") is True
