from __future__ import annotations

from fastapi import APIRouter, Depends

from app.domain.ports import ComparacionHistoricaRepositoryPort, DistritoRepositoryPort
from app.infrastructure.web.dependencias import get_comparacion_historica, get_distrito_repo
from app.infrastructure.web.schemas import ComparacionDistritoOut, ComparacionHistoricaResponse

router = APIRouter(prefix="/api/v1/comparacion-historica", tags=["impacto"])


@router.get("", response_model=ComparacionHistoricaResponse)
def comparacion_historica(
    comparacion: ComparacionHistoricaRepositoryPort = Depends(get_comparacion_historica),
    distrito_repo: DistritoRepositoryPort = Depends(get_distrito_repo),
) -> ComparacionHistoricaResponse:
    items = []
    for c in comparacion.por_distrito():
        distrito = distrito_repo.obtener(c.distrito_codigo)
        items.append(
            ComparacionDistritoOut(
                distrito_codigo=c.distrito_codigo,
                nombre=distrito.nombre if distrito else c.distrito_codigo,
                observado=c.observado,
                predicho=c.predicho,
                brecha_pct=c.brecha_pct,
            )
        )
    items.sort(key=lambda i: i.observado, reverse=True)
    return ComparacionHistoricaResponse(distritos=items)
