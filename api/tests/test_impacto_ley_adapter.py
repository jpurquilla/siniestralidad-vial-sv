"""Tests del adaptador de impacto de la Ley Cero Tolerancia (NB06). Todo el
contenido es real (nb06_resultados.json, ya generado y verificado por NB06);
este adaptador solo lee y da forma, no calcula nada nuevo."""

from __future__ import annotations

from datetime import date

import pytest

from app.config import settings
from app.infrastructure.analitica.impacto_ley_adapter import ImpactoLeyAdapter


@pytest.fixture
def adapter() -> ImpactoLeyAdapter:
    a = ImpactoLeyAdapter(settings.ruta_resultados_nb06, settings.ruta_parque_vehicular)
    a.cargar_y_validar()
    return a


class TestSerieAnual:
    def test_tiene_las_cinco_filas_oficiales(self, adapter):
        serie = adapter.serie_anual()
        assert len(serie) == 5
        assert [f.anio for f in serie] == ["2022", "2023", "2024", "2025", "2026"]

    def test_valores_2025_son_los_oficiales(self, adapter):
        fila_2025 = next(f for f in adapter.serie_anual() if f.anio == "2025")
        assert fila_2025.siniestros == 22265
        assert fila_2025.fallecidos == 1238
        assert fila_2025.letalidad == pytest.approx(1238 / 22265 * 100, abs=1e-6)

    def test_letalidad_baja_entre_2024_y_2025(self, adapter):
        serie = {f.anio: f for f in adapter.serie_anual()}
        assert serie["2025"].letalidad < serie["2024"].letalidad


class TestLey:
    def test_fecha_de_aprobacion(self, adapter):
        assert adapter.fecha_aprobacion_ley() == date(2024, 12, 20)


class TestResumen:
    def test_incluye_variacion_interanual_de_siniestros_2025(self, adapter):
        resumen = adapter.resumen()
        fila = next(r for r in resumen if r.metrica == "siniestros" and r.periodo == "2025")
        assert fila.variacion_interanual_pct == pytest.approx(9.674400275848472, abs=1e-6)

    def test_valores_nan_del_json_se_exponen_como_none(self, adapter):
        resumen = adapter.resumen()
        fila = next(r for r in resumen if r.metrica == "siniestros" and r.periodo == "2026 ene-jun")
        assert fila.variacion_interanual_pct is None


class TestLimitaciones:
    def test_expone_las_limitaciones_declaradas(self, adapter):
        limitaciones = adapter.limitaciones()
        assert len(limitaciones) == 5
        assert any("sintético" in l for l in limitaciones)

    def test_limitaciones_normalizadas_con_mayuscula_y_punto_final(self, adapter):
        for limitacion in adapter.limitaciones():
            assert not limitacion[0].islower()
            assert limitacion.endswith(".")

    def test_ya_no_dice_que_el_parque_no_esta_medido(self, adapter):
        limitaciones = adapter.limitaciones()
        assert not any("no medido" in l for l in limitaciones)
        assert any("parque vehicular real" in l for l in limitaciones)


class TestTasasPorVehiculo:
    """Parque real al 31-dic (suma de todas las clases y distritos del
    registro ONASEVI) contra la serie oficial: valores esperados del encargo."""

    def test_siniestros_por_mil_y_fallecidos_por_diez_mil(self, adapter):
        tasas = {t.anio: t for t in adapter.tasas_por_vehiculo()}
        assert tasas["2022"].siniestros_por_1000_vehiculos == pytest.approx(10.9, abs=0.05)
        assert tasas["2025"].siniestros_por_1000_vehiculos == pytest.approx(11.0, abs=0.05)
        assert tasas["2022"].fallecidos_por_10000_vehiculos == pytest.approx(8.44, abs=0.01)
        assert tasas["2025"].fallecidos_por_10000_vehiculos == pytest.approx(6.13, abs=0.01)

    def test_corte_es_31_de_diciembre_y_2026_es_parcial(self, adapter):
        tasas = {t.anio: t for t in adapter.tasas_por_vehiculo()}
        assert tasas["2022"].fecha_corte_parque == date(2022, 12, 31)
        assert tasas["2022"].parcial is False
        assert tasas["2026"].parcial is True
        assert tasas["2026"].fecha_corte_parque == date(2026, 6, 30)

    def test_corte_parcial_sale_de_la_cobertura_declarada(self):
        from app.infrastructure.analitica.impacto_ley_adapter import (
            ImpactoLeyRepositoryError,
            _corte_de_cobertura_parcial,
        )

        assert _corte_de_cobertura_parcial(2026, "parcial (ene-jun)") == date(2026, 6, 30)
        assert _corte_de_cobertura_parcial(2026, "parcial (ene-dic)") == date(2026, 12, 31)
        with pytest.raises(ImpactoLeyRepositoryError):
            _corte_de_cobertura_parcial(2026, "parcial")


class TestArranqueFallido:
    def test_falla_si_no_existe_el_archivo(self, tmp_path):
        a = ImpactoLeyAdapter(tmp_path / "no_existe.json", settings.ruta_parque_vehicular)
        with pytest.raises(Exception):
            a.cargar_y_validar()
