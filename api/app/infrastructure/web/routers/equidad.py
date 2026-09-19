from __future__ import annotations

from fastapi import APIRouter, Depends

from app.application.calcular_equidad import CalcularEquidad
from app.infrastructure.web.dependencias import get_calcular_equidad_uc
from app.infrastructure.web.schemas import EquidadResponse, GrupoEquidadOut

router = APIRouter(prefix="/api/v1/equidad", tags=["impacto"])


@router.get("", response_model=EquidadResponse)
def equidad(caso_uso: CalcularEquidad = Depends(get_calcular_equidad_uc)) -> EquidadResponse:
    resultado = caso_uso.ejecutar()
    return EquidadResponse(
        urbano_rural=[GrupoEquidadOut(**vars(g)) for g in resultado.urbano_rural],
        densidad_poblacional=[GrupoEquidadOut(**vars(g)) for g in resultado.densidad_poblacional],
    )
