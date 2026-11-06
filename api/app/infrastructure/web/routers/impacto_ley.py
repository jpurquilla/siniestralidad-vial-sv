from __future__ import annotations

from fastapi import APIRouter, Depends

from app.infrastructure.analitica.impacto_ley_adapter import ImpactoLeyAdapter
from app.infrastructure.web.dependencias import get_impacto_ley_adapter
from app.infrastructure.web.schemas import (
    ImpactoAnualOut,
    ImpactoLeyResponse,
    ImpactoResumenItemOut,
    TasaPorVehiculoOut,
)

router = APIRouter(prefix="/api/v1/impacto-ley-cero-tolerancia", tags=["impacto"])


@router.get("", response_model=ImpactoLeyResponse)
def impacto_ley(adapter: ImpactoLeyAdapter = Depends(get_impacto_ley_adapter)) -> ImpactoLeyResponse:
    return ImpactoLeyResponse(
        fecha_aprobacion_ley=adapter.fecha_aprobacion_ley(),
        serie_anual=[ImpactoAnualOut(**vars(f)) for f in adapter.serie_anual()],
        tasas_por_vehiculo=[TasaPorVehiculoOut(**vars(t)) for t in adapter.tasas_por_vehiculo()],
        resumen=[ImpactoResumenItemOut(**vars(r)) for r in adapter.resumen()],
        limitaciones=adapter.limitaciones(),
    )
