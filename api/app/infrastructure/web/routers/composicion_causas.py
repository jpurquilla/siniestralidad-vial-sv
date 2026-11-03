from __future__ import annotations

from fastapi import APIRouter, Depends

from app.domain.models import ComposicionCausas
from app.infrastructure.web.dependencias import get_composicion_causas
from app.infrastructure.web.schemas import (
    ComposicionCausasResponse,
    ComposicionFranjaOut,
    ProporcionCausaOut,
)

router = APIRouter(prefix="/api/v1/composicion-causas", tags=["causas"])


@router.get("", response_model=ComposicionCausasResponse)
def composicion_causas(
    composicion: ComposicionCausas = Depends(get_composicion_causas),
) -> ComposicionCausasResponse:
    return ComposicionCausasResponse(
        periodo_inicio=composicion.periodo_inicio,
        periodo_fin=composicion.periodo_fin,
        franjas=[
            ComposicionFranjaOut(
                franja=franja,
                n_siniestros=composicion.n_siniestros[franja],
                causas=[ProporcionCausaOut(causa=p.causa, proporcion=p.proporcion) for p in proporciones],
            )
            for franja, proporciones in composicion.por_franja.items()
        ],
    )
