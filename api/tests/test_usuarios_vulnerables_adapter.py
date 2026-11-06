"""Tests del adaptador de usuarios vulnerables (NB09 §4, retrospectivo).

Agrega matriz_severidad.csv (89,946 siniestros reales del conjunto sintético,
uno por fila). La metodología (letalidad = siniestros con al menos un
fallecido / total de siniestros del grupo) se verificó reproduciendo a mano
los números ya publicados en NB09_RESUMEN.md antes de escribir el adaptador:
Peatón 42.47% / 0.2294, Motociclista 36.94% / 0.1500, edad 71+ → 0.2982,
edad 0-15 → 0.0900. Estos tests fijan esos mismos números como contrato."""

from __future__ import annotations

import pytest

from app.config import settings
from app.infrastructure.analitica.usuarios_vulnerables_adapter import (
    UsuariosVulnerablesAdapter,
)


@pytest.fixture(scope="module")
def adapter() -> UsuariosVulnerablesAdapter:
    a = UsuariosVulnerablesAdapter(settings.ruta_matriz_severidad)
    a.cargar_y_validar()
    return a


class TestRepartoPorTipo:
    def test_excluye_sin_victimas(self, adapter):
        tipos = {r.tipo for r in adapter.reparto_por_tipo()}
        assert "Sin victimas" not in tipos
        assert tipos == {"Peatón", "Motociclista", "Ciclista", "Otro"}

    def test_peaton_coincide_con_nb09_resumen(self, adapter):
        fila = next(r for r in adapter.reparto_por_tipo() if r.tipo == "Peatón")
        assert fila.pct_fatales == pytest.approx(42.47, abs=0.01)
        assert fila.letalidad_por_siniestro == pytest.approx(0.2294, abs=0.0001)

    def test_motociclista_coincide_con_nb09_resumen(self, adapter):
        fila = next(r for r in adapter.reparto_por_tipo() if r.tipo == "Motociclista")
        assert fila.pct_fatales == pytest.approx(36.94, abs=0.01)
        assert fila.letalidad_por_siniestro == pytest.approx(0.1500, abs=0.0001)

    def test_los_porcentajes_suman_cien(self, adapter):
        total = sum(r.pct_fatales for r in adapter.reparto_por_tipo())
        assert total == pytest.approx(100.0, abs=0.01)


class TestLetalidadPorEdad:
    def test_maximo_en_71_mas(self, adapter):
        fila = next(r for r in adapter.letalidad_por_edad() if r.rango_etario == "71+")
        assert fila.letalidad == pytest.approx(0.2982, abs=0.0001)

    def test_minimo_en_0_15(self, adapter):
        fila = next(r for r in adapter.letalidad_por_edad() if r.rango_etario == "0-15")
        assert fila.letalidad == pytest.approx(0.0900, abs=0.0001)

    def test_trece_rangos_etarios(self, adapter):
        assert len(adapter.letalidad_por_edad()) == 13


class TestArranqueFallido:
    def test_falla_si_no_existe_el_archivo(self, tmp_path):
        a = UsuariosVulnerablesAdapter(tmp_path / "no_existe.csv")
        with pytest.raises(Exception):
            a.cargar_y_validar()
