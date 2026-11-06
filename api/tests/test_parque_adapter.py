"""Tests del adaptador de crecimiento del parque vehicular (crec_parque_12m)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.config import settings
from app.domain.models import FuenteParque
from app.infrastructure.modelo.parque_adapter import ParqueAdapter, ParqueRepositoryError


@pytest.fixture(scope="module")
def adapter() -> ParqueAdapter:
    a = ParqueAdapter(settings.ruta_matriz_frecuencia)
    a.cargar_y_validar()
    return a


class TestDentroDelRango:
    def test_san_salvador_2024_06_es_observado_y_coincide_con_la_matriz(self, adapter):
        valor, fuente = adapter.crec_parque_12m("san salvador", date(2024, 6, 15))
        assert fuente == FuenteParque.OBSERVADO
        assert valor == pytest.approx(0.043012, abs=1e-6)

    def test_limites_del_rango_son_observados(self, adapter):
        assert adapter.mes_minimo == "2022-01"
        assert adapter.mes_maximo == "2026-06"
        assert adapter.crec_parque_12m("acajutla", date(2022, 1, 1))[1] == FuenteParque.OBSERVADO
        assert adapter.crec_parque_12m("acajutla", date(2026, 6, 30))[1] == FuenteParque.OBSERVADO


class TestFueraDelRango:
    def test_fecha_futura_usa_el_ultimo_mes_disponible(self, adapter):
        ultimo, _ = adapter.crec_parque_12m("acajutla", date(2026, 6, 30))
        valor, fuente = adapter.crec_parque_12m("acajutla", date(2027, 3, 10))
        assert fuente == FuenteParque.ULTIMO_DISPONIBLE
        assert valor == ultimo

    def test_fecha_anterior_usa_el_mes_disponible_mas_cercano(self, adapter):
        primero, _ = adapter.crec_parque_12m("acajutla", date(2022, 1, 1))
        valor, fuente = adapter.crec_parque_12m("acajutla", date(2021, 5, 1))
        assert fuente == FuenteParque.ULTIMO_DISPONIBLE
        assert valor == primero

    def test_distrito_desconocido_lanza_error(self, adapter):
        with pytest.raises(ParqueRepositoryError):
            adapter.crec_parque_12m("no_existe", date(2024, 1, 1))


class TestArranqueFallido:
    def test_falla_si_no_existe_la_matriz(self, tmp_path):
        with pytest.raises(ParqueRepositoryError):
            ParqueAdapter(tmp_path / "no_existe.csv.gz").cargar_y_validar()

    def test_falla_si_crec_parque_no_es_unico_por_distrito_y_mes(self, tmp_path):
        df = pd.DataFrame(
            {
                "distrito_key": ["a", "a"],
                "anio_mes": ["2024-01", "2024-01"],
                "crec_parque_12m": [0.1, 0.2],
            }
        )
        ruta = tmp_path / "matriz.csv"
        df.to_csv(ruta, index=False)
        with pytest.raises(ParqueRepositoryError, match="único"):
            ParqueAdapter(ruta).cargar_y_validar()

    def test_falla_si_faltan_columnas(self, tmp_path):
        ruta = tmp_path / "matriz.csv"
        pd.DataFrame({"distrito_key": ["a"]}).to_csv(ruta, index=False)
        with pytest.raises(ParqueRepositoryError, match="columnas"):
            ParqueAdapter(ruta).cargar_y_validar()
