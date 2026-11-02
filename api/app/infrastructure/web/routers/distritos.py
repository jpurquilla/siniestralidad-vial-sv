from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.domain.ports import DistritoRepositoryPort
from app.infrastructure.geo.geojson_adapter import GeoJsonAdapter
from app.infrastructure.web.dependencias import get_distrito_repo, get_geojson_adapter
from app.infrastructure.web.schemas import DistritoOut, DistritosResponse

router = APIRouter(prefix="/api/v1/distritos", tags=["distritos"])


@router.get("", response_model=DistritosResponse)
def listar_distritos(repo: DistritoRepositoryPort = Depends(get_distrito_repo)) -> DistritosResponse:
    distritos = [DistritoOut(**d.__dict__) for d in repo.listar()]
    return DistritosResponse(total=len(distritos), distritos=distritos)


@router.get("/geojson")
def geojson_distritos(adapter: GeoJsonAdapter = Depends(get_geojson_adapter)) -> dict:
    geojson = adapter.obtener_geojson()
    if geojson is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No hay geometrías de distritos disponibles. Coloque el archivo "
                f"GeoJSON en {settings.ruta_geojson_distritos} (ver README, sección "
                "'GeoJSON de distritos') para habilitar este endpoint."
            ),
        )
    return geojson
