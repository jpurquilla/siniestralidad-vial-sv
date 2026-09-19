from __future__ import annotations

from fastapi import APIRouter, Depends

from app.infrastructure.analitica.usuarios_vulnerables_adapter import (
    UsuariosVulnerablesAdapter,
)
from app.infrastructure.web.dependencias import get_usuarios_vulnerables_adapter
from app.infrastructure.web.schemas import (
    LetalidadPorEdadOut,
    RepartoTipoUsuarioOut,
    UsuariosVulnerablesResponse,
)

router = APIRouter(prefix="/api/v1/usuarios-vulnerables", tags=["impacto"])


@router.get("", response_model=UsuariosVulnerablesResponse)
def usuarios_vulnerables(
    adapter: UsuariosVulnerablesAdapter = Depends(get_usuarios_vulnerables_adapter),
) -> UsuariosVulnerablesResponse:
    return UsuariosVulnerablesResponse(
        reparto_por_tipo=[RepartoTipoUsuarioOut(**vars(r)) for r in adapter.reparto_por_tipo()],
        letalidad_por_edad=[LetalidadPorEdadOut(**vars(l)) for l in adapter.letalidad_por_edad()],
    )
