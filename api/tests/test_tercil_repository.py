from __future__ import annotations

import json

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
    datos = json.loads(settings.ruta_terciles.read_text(encoding="utf-8"))
    assert confiabilidad == pytest.approx(datos["confiabilidad_por_tercil"]["alto"]["spearman"])


def test_el_archivo_de_terciles_es_el_del_modelo_final():
    assert settings.ruta_terciles.name == "terciles_nb12.json"
    datos = json.loads(settings.ruta_terciles.read_text(encoding="utf-8"))
    conteo = {t: list(datos["tercil_por_distrito"].values()).count(t) for t in ("alto", "medio", "bajo")}
    assert conteo == {"alto": 35, "medio": 34, "bajo": 34}


def test_distrito_desconocido_lanza_error(repo):
    with pytest.raises(TercilRepositoryError):
        repo.tercil_y_confiabilidad("no_existe")
