from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException

from app.application.predecir_celda import DistritoNoEncontradoError, PeticionPrediccionCelda, PredecirCelda
from app.infrastructure.web.dependencias import get_predecir_celda_uc
from app.infrastructure.web.schemas import CausaEsperadaOut, PrediccionCeldaOut, PrediccionRequest, PrediccionResponse

router = APIRouter(prefix="/api/v1/predicciones", tags=["predicciones"])


def _fechas_en_rango(inicio, fin):
    dias = (fin - inicio).days
    for i in range(dias + 1):
        yield inicio + timedelta(days=i)


@router.post("", response_model=PrediccionResponse)
def predecir(
    peticion: PrediccionRequest,
    caso_uso: PredecirCelda = Depends(get_predecir_celda_uc),
) -> PrediccionResponse:
    # Se construyen todas las celdas y se predice en UNA sola llamada
    # vectorizada al modelo (un mes completo son ~12,000 celdas).
    fechas = list(_fechas_en_rango(peticion.fecha_inicio, peticion.fecha_fin))
    sub_peticiones = [
        PeticionPrediccionCelda(
            distrito_codigo=distrito_codigo,
            fecha=fecha,
            franja=franja,
            prcp_mensual=peticion.prcp_mensual,
            factor_prcp=peticion.factor_prcp,
        )
        for distrito_codigo in peticion.distritos
        for fecha in fechas
        for franja in peticion.franjas
    ]
    try:
        resultados = caso_uso.ejecutar_lote(sub_peticiones)
    except DistritoNoEncontradoError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return PrediccionResponse(
        predicciones=[
            PrediccionCeldaOut(
                distrito_codigo=r.distrito_codigo,
                fecha=r.fecha,
                franja=r.franja,
                siniestros_esperados=r.siniestros_esperados,
                fuente_prcp=r.fuente_prcp,
                fuente_parque=r.fuente_parque,
                prcp_mensual=r.prcp_mensual,
                por_causa=[CausaEsperadaOut(**vars(c)) for c in r.por_causa],
            )
            for r in resultados
        ]
    )
