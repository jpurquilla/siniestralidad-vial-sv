"""Factorías de dependencias FastAPI: leen los singletons construidos en el
lifespan (app.state) — nunca variables globales de módulo."""

from __future__ import annotations

from fastapi import Request

from app.application.calcular_equidad import CalcularEquidad
from app.application.predecir_celda import PredecirCelda
from app.application.ranking_distritos import RankingDistritos
from app.domain.ports import (
    ComparacionHistoricaRepositoryPort,
    DistritoRepositoryPort,
    SiniestroPuntoRepositoryPort,
)
from app.infrastructure.analitica.impacto_ley_adapter import ImpactoLeyAdapter
from app.infrastructure.analitica.usuarios_vulnerables_adapter import (
    UsuariosVulnerablesAdapter,
)
from app.infrastructure.geo.geojson_adapter import GeoJsonAdapter


def get_predecir_celda_uc(request: Request) -> PredecirCelda:
    return request.app.state.predecir_celda_uc


def get_ranking_uc(request: Request) -> RankingDistritos:
    return request.app.state.ranking_uc


def get_distrito_repo(request: Request) -> DistritoRepositoryPort:
    return request.app.state.distrito_repo


def get_geojson_adapter(request: Request) -> GeoJsonAdapter:
    return request.app.state.geojson_adapter


def get_info_modelo(request: Request) -> dict:
    return request.app.state.info_modelo


def get_estado_salud(request: Request) -> dict:
    return request.app.state.estado_salud


def get_impacto_ley_adapter(request: Request) -> ImpactoLeyAdapter:
    return request.app.state.impacto_ley_adapter


def get_usuarios_vulnerables_adapter(request: Request) -> UsuariosVulnerablesAdapter:
    return request.app.state.usuarios_vulnerables_adapter


def get_comparacion_historica(request: Request) -> ComparacionHistoricaRepositoryPort:
    return request.app.state.comparacion_historica_adapter


def get_calcular_equidad_uc(request: Request) -> CalcularEquidad:
    return request.app.state.calcular_equidad_uc


def get_siniestro_punto_repo(request: Request) -> SiniestroPuntoRepositoryPort:
    return request.app.state.siniestro_punto_repo
