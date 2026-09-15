"""Factorías de dependencias FastAPI: leen los singletons construidos en el
lifespan (app.state) — nunca variables globales de módulo."""

from __future__ import annotations

from fastapi import Request

from app.application.predecir_celda import PredecirCelda
from app.application.ranking_distritos import RankingDistritos
from app.domain.ports import DistritoRepositoryPort
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
