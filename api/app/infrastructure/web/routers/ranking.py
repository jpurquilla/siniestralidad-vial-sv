from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from app.application.ranking_distritos import PeticionRanking, RankingDistritos
from app.domain.models import Franja
from app.domain.ports import DistritoRepositoryPort
from app.infrastructure.web.dependencias import get_distrito_repo, get_ranking_uc
from app.infrastructure.web.schemas import RankingItemOut, RankingQuery, RankingResponse

router = APIRouter(prefix="/api/v1/ranking", tags=["ranking"])


@router.get("", response_model=RankingResponse)
def ranking(
    fecha_inicio: date = Query(...),
    fecha_fin: date = Query(...),
    franjas: list[Franja] | None = Query(default=None),
    caso_uso: RankingDistritos = Depends(get_ranking_uc),
    distrito_repo: DistritoRepositoryPort = Depends(get_distrito_repo),
) -> RankingResponse:
    try:
        query = RankingQuery(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            **({"franjas": franjas} if franjas else {}),
        )
    except ValidationError as exc:
        errores = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        raise HTTPException(status_code=422, detail=errores) from exc

    peticion = PeticionRanking(
        fecha_inicio=query.fecha_inicio, fecha_fin=query.fecha_fin, franjas=tuple(query.franjas)
    )
    resultado = caso_uso.ejecutar(peticion)

    items = []
    for r in resultado:
        distrito = distrito_repo.obtener(r.distrito_codigo)
        items.append(
            RankingItemOut(
                distrito_codigo=r.distrito_codigo,
                nombre=distrito.nombre if distrito else r.distrito_codigo,
                siniestros_esperados=r.siniestros_esperados,
                tercil=r.tercil,
                confiabilidad_spearman=r.confiabilidad_spearman,
            )
        )

    return RankingResponse(fecha_inicio=query.fecha_inicio, fecha_fin=query.fecha_fin, ranking=items)
