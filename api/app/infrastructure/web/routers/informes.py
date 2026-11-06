from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import ValidationError

from app.application.generar_informe_periodo import (
    GenerarInformePeriodo,
    PeticionInformeInvalidaError,
    PeticionInformePeriodo,
)
from app.domain.informe import InformePeriodo
from app.domain.models import Franja
from app.infrastructure.informes.renderer_excel import RendererExcel
from app.infrastructure.informes.renderer_pdf import RendererPdf
from app.infrastructure.web.dependencias import (
    get_generar_informe_periodo_uc,
    get_renderer_excel,
    get_renderer_pdf,
)
from app.infrastructure.web.schemas_informe import InformePeriodoRequest, InformePeriodoResponse

router = APIRouter(prefix="/api/v1/informes", tags=["informes"])

_MEDIA_PDF = "application/pdf"
_MEDIA_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _generar(caso_uso: GenerarInformePeriodo, p: InformePeriodoRequest) -> InformePeriodo:
    try:
        return caso_uso.ejecutar(
            PeticionInformePeriodo(
                fecha_inicio=p.fecha_inicio,
                fecha_fin=p.fecha_fin,
                departamento=p.departamento,
                distritos=tuple(p.distritos),
                franjas=tuple(p.franjas),
                dias_semana=tuple(p.dias_semana),
            )
        )
    except PeticionInformeInvalidaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _peticion_desde_query(
    fecha_inicio: date = Query(description="Primer día del periodo (YYYY-MM-DD)."),
    fecha_fin: date = Query(description="Último día del periodo, inclusive (máximo 92 días)."),
    departamento: str | None = Query(default=None, description="Nombre con tildes."),
    distritos: list[str] = Query(default=[], description="Se repite: distritos=a&distritos=b."),
    franjas: list[Franja] = Query(default=[], description="Se repite: franjas=...&franjas=..."),
    dias_semana: list[int] = Query(default=[], description="0 = lunes ... 6 = domingo. Se repite."),
) -> InformePeriodoRequest:
    try:
        return InformePeriodoRequest(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            departamento=departamento,
            distritos=distritos,
            franjas=franjas,
            dias_semana=dias_semana,
        )
    except ValidationError as exc:
        mensajes = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())
        raise HTTPException(status_code=422, detail=mensajes) from exc


def _adjunto(contenido: bytes, media_type: str, p: InformePeriodoRequest, extension: str) -> Response:
    nombre = f"informe-siniestros-{p.fecha_inicio.isoformat()}_{p.fecha_fin.isoformat()}.{extension}"
    return Response(
        content=contenido,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )


@router.post("/periodo", response_model=InformePeriodoResponse)
def informe_periodo(
    peticion: InformePeriodoRequest,
    caso_uso: GenerarInformePeriodo = Depends(get_generar_informe_periodo_uc),
) -> InformePeriodoResponse:
    return InformePeriodoResponse.desde_dominio(_generar(caso_uso, peticion))


@router.get("/periodo.pdf", response_class=Response, responses={200: {"content": {_MEDIA_PDF: {}}}})
def informe_periodo_pdf(
    peticion: InformePeriodoRequest = Depends(_peticion_desde_query),
    caso_uso: GenerarInformePeriodo = Depends(get_generar_informe_periodo_uc),
    renderer: RendererPdf = Depends(get_renderer_pdf),
) -> Response:
    return _adjunto(renderer.renderizar(_generar(caso_uso, peticion)), _MEDIA_PDF, peticion, "pdf")


@router.get("/periodo.xlsx", response_class=Response, responses={200: {"content": {_MEDIA_XLSX: {}}}})
def informe_periodo_xlsx(
    peticion: InformePeriodoRequest = Depends(_peticion_desde_query),
    caso_uso: GenerarInformePeriodo = Depends(get_generar_informe_periodo_uc),
    renderer: RendererExcel = Depends(get_renderer_excel),
) -> Response:
    return _adjunto(renderer.renderizar(_generar(caso_uso, peticion)), _MEDIA_XLSX, peticion, "xlsx")
