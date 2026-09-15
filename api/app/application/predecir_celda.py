"""Caso de uso: predecir siniestros esperados para una celda distrito×fecha×franja.

Python puro: orquesta los puertos del dominio, sin importar FastAPI, pandas
ni xgboost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.domain import calendario
from app.domain.models import FeaturesCelda, Franja, FuentePrecipitacion, PrediccionCelda
from app.domain.ports import (
    CalendarioRepositoryPort,
    ClimaRepositoryPort,
    DistritoRepositoryPort,
    ExposicionRepositoryPort,
    PredictorFrecuenciaPort,
)


class DistritoNoEncontradoError(Exception):
    def __init__(self, codigo: str) -> None:
        self.codigo = codigo
        super().__init__(f"Distrito no encontrado: {codigo!r}")


@dataclass(frozen=True)
class PeticionPrediccionCelda:
    distrito_codigo: str
    fecha: date
    franja: Franja
    prcp_mensual: float | None = None  # None = usar normal climatológica


class PredecirCelda:
    def __init__(
        self,
        predictor: PredictorFrecuenciaPort,
        distritos: DistritoRepositoryPort,
        clima: ClimaRepositoryPort,
        calendario_repo: CalendarioRepositoryPort,
        exposicion: ExposicionRepositoryPort,
    ) -> None:
        self._predictor = predictor
        self._distritos = distritos
        self._clima = clima
        self._calendario = calendario_repo
        self._exposicion = exposicion

    def construir_features(self, peticion: PeticionPrediccionCelda) -> tuple[FeaturesCelda, FuentePrecipitacion]:
        distrito = self._distritos.obtener(peticion.distrito_codigo)
        if distrito is None:
            raise DistritoNoEncontradoError(peticion.distrito_codigo)

        if peticion.prcp_mensual is not None:
            prcp_mensual = peticion.prcp_mensual
            fuente_prcp = FuentePrecipitacion.OBSERVADA
        else:
            prcp_mensual = self._clima.normal_climatologica(distrito.codigo, peticion.fecha.month)
            fuente_prcp = FuentePrecipitacion.NORMAL_CLIMATOLOGICA

        log_exposicion_v2 = self._exposicion.log_exposicion_v2(distrito.codigo, peticion.fecha)
        es_feriado = self._calendario.es_feriado(peticion.fecha, distrito.codigo)

        features = FeaturesCelda(
            es_lluviosa=calendario.es_lluviosa(peticion.fecha),
            es_finde=calendario.es_finde(peticion.fecha),
            dia_semana=calendario.dia_semana(peticion.fecha),
            periodo_agostino=calendario.periodo_agostino(peticion.fecha),
            es_feriado=int(es_feriado),
            prcp_mensual=prcp_mensual,
            pct_urbano=distrito.pct_urbano,
            poblacion_baja=int(distrito.poblacion_baja),
            log_exposicion_v2=log_exposicion_v2,
        )
        return features, fuente_prcp

    def ejecutar(self, peticion: PeticionPrediccionCelda) -> PrediccionCelda:
        features, fuente_prcp = self.construir_features(peticion)
        (siniestros_esperados,) = self._predictor.predecir_lote([features])
        return PrediccionCelda(
            distrito_codigo=peticion.distrito_codigo,
            fecha=peticion.fecha,
            franja=peticion.franja,
            siniestros_esperados=siniestros_esperados,
            fuente_prcp=fuente_prcp,
        )
