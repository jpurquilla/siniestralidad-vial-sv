"""Tests del catálogo de distritos y de la normal climatológica."""

from __future__ import annotations

import pytest

from app.config import settings
from app.infrastructure.geo.distrito_repository import DistritoRepository
from app.infrastructure.modelo.clima_adapter import ClimaAdapter


@pytest.fixture(scope="module")
def repo() -> DistritoRepository:
    r = DistritoRepository(settings.ruta_matriz_frecuencia, settings.ruta_siniestros)
    r.cargar_y_validar()
    return r


class TestDistritoRepository:
    def test_lista_103_distritos(self, repo):
        assert len(repo.listar()) == 103

    def test_san_salvador_existe_con_datos_esperados(self, repo):
        d = repo.obtener("san salvador")
        assert d is not None
        assert d.nombre == "San Salvador"
        assert d.pct_urbano == pytest.approx(0.9965, abs=1e-4)
        assert d.poblacion_baja is False

    def test_distrito_inexistente_devuelve_none(self, repo):
        assert repo.obtener("no_existe") is None

    def test_existe(self, repo):
        assert repo.existe("san salvador") is True
        assert repo.existe("no_existe") is False

    def test_dolores_se_reconcilia_a_villa_dolores(self, repo):
        # MAPEO_DISTRITOS: 'dolores' en siniestros.csv -> 'villa dolores'
        assert repo.existe("villa dolores")


@pytest.fixture(scope="module")
def clima() -> ClimaAdapter:
    c = ClimaAdapter(settings.ruta_matriz_frecuencia)
    c.cargar_y_validar()
    return c


class TestClimaAdapter:
    def test_normal_climatologica_positiva(self, clima):
        valor = clima.normal_climatologica("acajutla", 6)
        assert valor > 0

    def test_distrito_desconocido_lanza_error(self, clima):
        from app.infrastructure.modelo.clima_adapter import ClimaRepositoryError

        with pytest.raises(ClimaRepositoryError):
            clima.normal_climatologica("no_existe", 6)
