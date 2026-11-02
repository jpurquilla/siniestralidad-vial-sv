from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException

from app.application.predecir_celda import DistritoNoEncontradoError, PeticionPrediccionCelda, PredecirCelda
from app.infrastructure.web.dependencias import get_predecir_celda_uc
from app.infrastructure.web.schemas import PrediccionCeldaOut, PrediccionRequest, PrediccionResponse

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
    predicciones: list[PrediccionCeldaOut] = []

    for distrito_codigo in peticion.distritos:
        for fecha in _fechas_en_rango(peticion.fecha_inicio, peticion.fecha_fin):
            for franja in peticion.franjas:
                sub_peticion = PeticionPrediccionCelda(
                    distrito_codigo=distrito_codigo,
                    fecha=fecha,
                    franja=franja,
                    prcp_mensual=peticion.prcp_mensual,
                )
                try:
                    resultado = caso_uso.ejecutar(sub_peticion)
                except DistritoNoEncontradoError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc

                predicciones.append(
                    PrediccionCeldaOut(
                        distrito_codigo=resultado.distrito_codigo,
                        fecha=resultado.fecha,
                        franja=resultado.franja,
                        siniestros_esperados=resultado.siniestros_esperados,
                        fuente_prcp=resultado.fuente_prcp,
                    )
                )

    return PrediccionResponse(predicciones=predicciones)
