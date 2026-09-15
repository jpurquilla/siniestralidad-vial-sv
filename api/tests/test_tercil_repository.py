from __future__ import annotations

import pytest

from app.config import settings
from app.domain.models import Tercil
from app.infrastructure.modelo.tercil_repository import TercilRepository, TercilRepositoryError


@pytest.fixture(scope="module")
def repo() -> TercilRepository:
    r = TercilRepository(settings.ruta_terciles)
    r.cargar_y_validar()
    return r


def test_san_salvador_es_tercil_alto(repo):
    tercil, confiabilidad = repo.tercil_y_confiabilidad("san salvador")
    assert tercil == Tercil.ALTO
    assert confiabilidad == pytest.approx(0.9775, abs=1e-3)


def test_distrito_desconocido_lanza_error(repo):
    with pytest.raises(TercilRepositoryError):
        repo.tercil_y_confiabilidad("no_existe")
