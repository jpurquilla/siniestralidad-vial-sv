from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.application.agregar_siniestros import AgregarSiniestros
from app.domain.models import FiltroSiniestros
from app.domain.ports import SiniestroPuntoRepositoryPort
from app.infrastructure.web.dependencias import get_agregar_siniestros_uc, get_siniestro_punto_repo
from app.infrastructure.web.schemas import (
    DistritoAgregadoOut,
    HexagonoOut,
    SiniestrosDistritoResponse,
    SiniestrosHexResponse,
    OpcionesFiltroSiniestrosResponse,
    SiniestroPuntoOut,
    SiniestrosPuntosResponse,
)

router = APIRouter(prefix="/api/v1/siniestros", tags=["hotspots"])

_MAX_PUNTOS_POR_RESPUESTA = 90_000


@router.get(
    "/puntos",
    response_model=SiniestrosPuntosResponse | SiniestrosHexResponse | SiniestrosDistritoResponse,
)
def puntos(
    fecha_inicio: date | None = Query(default=None),
    fecha_fin: date | None = Query(default=None),
    tipo_siniestro: str | None = Query(default=None),
    tipo_vehiculo: str | None = Query(default=None),
    factor_causante: str | None = Query(default=None),
    hora_inicio: str | None = Query(default=None, pattern=r"^\d{2}:\d{2}$"),
    hora_fin: str | None = Query(default=None, pattern=r"^\d{2}:\d{2}$"),
    departamento: str | None = Query(default=None),
    distrito: str | None = Query(default=None),
    agregacion: Literal["hex", "distrito"] | None = Query(
        default=None,
        description="Sin valor: puntos individuales. «hex»: hexágonos con su conteo. "
        "«distrito»: conteo, población y tasa por 10,000 habitantes de cada distrito. "
        "Las agregaciones respetan los mismos filtros y no tienen tope de respuesta.",
    ),
    repo: SiniestroPuntoRepositoryPort = Depends(get_siniestro_punto_repo),
    agregar: AgregarSiniestros = Depends(get_agregar_siniestros_uc),
) -> SiniestrosPuntosResponse | SiniestrosHexResponse | SiniestrosDistritoResponse:
    if fecha_inicio and fecha_fin and fecha_fin < fecha_inicio:
        raise HTTPException(422, detail="fecha_fin no puede ser anterior a fecha_inicio.")

    filtro = FiltroSiniestros(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        tipo_siniestro=tipo_siniestro,
        tipo_vehiculo=tipo_vehiculo,
        factor_causante=factor_causante,
        hora_inicio=hora_inicio,
        hora_fin=hora_fin,
        departamento=departamento,
        distrito=distrito,
    )
    if agregacion == "hex":
        hexagonos = agregar.por_hexagono(filtro)
        return SiniestrosHexResponse(
            lado_km=agregar.lado_hexagono_km,
            total=sum(h.conteo for h in hexagonos),
            hexagonos=[HexagonoOut(**vars(h)) for h in hexagonos],
        )
    if agregacion == "distrito":
        distritos = agregar.por_distrito(filtro)
        return SiniestrosDistritoResponse(
            total=sum(d.siniestros for d in distritos),
            distritos=[DistritoAgregadoOut(**vars(d)) for d in distritos],
        )

    resultado = repo.buscar(filtro)

    if len(resultado) > _MAX_PUNTOS_POR_RESPUESTA:
        raise HTTPException(
            413,
            detail=(
                f"La consulta devuelve {len(resultado)} puntos; el máximo por "
                f"respuesta es {_MAX_PUNTOS_POR_RESPUESTA}. Acotá con más filtros "
                "(fecha, departamento, tipo de siniestro)."
            ),
        )

    return SiniestrosPuntosResponse(
        total=len(resultado),
        puntos=[SiniestroPuntoOut(**vars(p)) for p in resultado],
    )


@router.get("/opciones-filtro", response_model=OpcionesFiltroSiniestrosResponse)
def opciones_filtro(
    repo: SiniestroPuntoRepositoryPort = Depends(get_siniestro_punto_repo),
) -> OpcionesFiltroSiniestrosResponse:
    return OpcionesFiltroSiniestrosResponse(**vars(repo.opciones_filtro()))
