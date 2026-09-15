from __future__ import annotations

from fastapi import APIRouter, Depends

from app.infrastructure.web.dependencias import get_estado_salud
from app.infrastructure.web.schemas import HealthResponse

router = APIRouter(tags=["salud"])


@router.get("/health", response_model=HealthResponse)
def health(estado: dict = Depends(get_estado_salud)) -> HealthResponse:
    return HealthResponse(**estado)
