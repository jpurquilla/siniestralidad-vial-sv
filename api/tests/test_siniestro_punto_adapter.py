"""Tests del adaptador de puntos individuales de siniestros (mapa de
hotspots, replica de observatoriovial.fonat.gob.sv). Real: coordenadas GPS
de `siniestros.csv` (conjunto sintético calibrado, ver NB01)."""

from __future__ import annotations

from datetime import date

import pytest

from app.config import settings
from app.domain.models import FiltroSiniestros
from app.infrastructure.analitica.siniestro_punto_adapter import SiniestroPuntoAdapter


@pytest.fixture(scope="module")
def adapter() -> SiniestroPuntoAdapter:
    a = SiniestroPuntoAdapter(settings.ruta_siniestros)
    a.cargar_y_validar()
    return a


class TestSinFiltro:
    def test_devuelve_los_89946_puntos(self, adapter):
        assert len(adapter.buscar(FiltroSiniestros())) == 89_946

    def test_cada_punto_tiene_coordenadas_reales_de_el_salvador(self, adapter):
        # Bounding box real de El Salvador (con margen).
        for p in adapter.buscar(FiltroSiniestros())[:200]:
            assert 12.9 <= p.latitud <= 14.5
            assert -90.2 <= p.longitud <= -87.6


class TestFiltroPorFecha:
    def test_filtra_por_rango_de_fechas(self, adapter):
        resultado = adapter.buscar(
            FiltroSiniestros(fecha_inicio=date(2022, 1, 1), fecha_fin=date(2022, 1, 1))
        )
        assert len(resultado) > 0
        assert all(p.fecha == date(2022, 1, 1) for p in resultado)


class TestFiltroPorCategoria:
    def test_filtra_por_tipo_siniestro(self, adapter):
        resultado = adapter.buscar(FiltroSiniestros(tipo_siniestro="Volcamiento"))
        assert len(resultado) == 1656
        assert all(p.tipo_siniestro == "Volcamiento" for p in resultado)

    def test_filtra_por_departamento(self, adapter):
        resultado = adapter.buscar(FiltroSiniestros(departamento="Cabañas"))
        assert len(resultado) > 0
        assert all(p.departamento == "Cabañas" for p in resultado)

    def test_combina_varios_filtros_con_and(self, adapter):
        solo_tipo = adapter.buscar(FiltroSiniestros(tipo_siniestro="Atropello"))
        combinado = adapter.buscar(
            FiltroSiniestros(tipo_siniestro="Atropello", departamento="San Salvador")
        )
        assert 0 < len(combinado) < len(solo_tipo)


class TestFiltroPorHora:
    def test_filtra_por_rango_horario(self, adapter):
        resultado = adapter.buscar(FiltroSiniestros(hora_inicio="00:00", hora_fin="05:59"))
        assert len(resultado) > 0
        assert all(p.hora <= "05:59" for p in resultado)


class TestOpcionesFiltro:
    def test_incluye_los_14_departamentos_reales(self, adapter):
        opciones = adapter.opciones_filtro()
        assert len(opciones.departamentos) == 14
        assert "San Salvador" in opciones.departamentos

    def test_incluye_los_5_tipos_de_siniestro_reales(self, adapter):
        opciones = adapter.opciones_filtro()
        assert set(opciones.tipos_siniestro) == {
            "Colision", "Atropello", "Caracteristicas especiales",
            "Choque contra objeto fijo", "Volcamiento",
        }


class TestArranqueFallido:
    def test_falla_si_no_existe_el_archivo(self, tmp_path):
        a = SiniestroPuntoAdapter(tmp_path / "no_existe.csv")
        with pytest.raises(Exception):
            a.cargar_y_validar()
