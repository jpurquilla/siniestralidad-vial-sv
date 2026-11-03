from __future__ import annotations

from fastapi import APIRouter, Depends

from app.domain.ports import ClimaRepositoryPort
from app.infrastructure.modelo.clima_adapter import ANIO_ENTRENAMIENTO_FIN, ANIO_ENTRENAMIENTO_INICIO
from app.infrastructure.web.dependencias import get_clima_repo
from app.infrastructure.web.schemas import RangoPrecipitacionOut

router = APIRouter(prefix="/api/v1/clima", tags=["clima"])


@router.get("/rango-entrenamiento", response_model=RangoPrecipitacionOut)
def rango_entrenamiento(clima: ClimaRepositoryPort = Depends(get_clima_repo)) -> RangoPrecipitacionOut:
    minimo, maximo = clima.rango_entrenamiento()
    return RangoPrecipitacionOut(
        minimo_mm=minimo,
        maximo_mm=maximo,
        anio_inicio=ANIO_ENTRENAMIENTO_INICIO,
        anio_fin=ANIO_ENTRENAMIENTO_FIN,
    )
