from __future__ import annotations

from fastapi import APIRouter, Depends

from app.infrastructure.web.dependencias import get_info_modelo
from app.infrastructure.web.schemas import ModeloInfoResponse

router = APIRouter(prefix="/api/v1/modelo", tags=["modelo"])


@router.get("/info", response_model=ModeloInfoResponse)
def info(info_modelo: dict = Depends(get_info_modelo)) -> ModeloInfoResponse:
    return ModeloInfoResponse(**info_modelo)
