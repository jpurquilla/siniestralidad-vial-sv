"""Caso de uso: ranking de distritos por siniestros esperados en un rango de
fechas, con su tercil (protocolo NB07 §5.4, calculado con el modelo final NB12) y
la confiabilidad descriptiva de ese ordenamiento.

Python puro: orquesta los puertos del dominio, sin importar FastAPI, pandas
ni xgboost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.domain import calendario
from app.domain.models import Distrito, FeaturesCelda, Franja, RankingDistrito
from app.domain.ports import (
    CalendarioRepositoryPort,
    ClimaRepositoryPort,
    DistritoRepositoryPort,
    ExposicionRepositoryPort,
    ParqueRepositoryPort,
    PredictorFrecuenciaPort,
    TercilRepositoryPort,
)

TODAS_LAS_FRANJAS: tuple[Franja, ...] = tuple(Franja)


@dataclass(frozen=True)
class PeticionRanking:
    fecha_inicio: date
    fecha_fin: date
    franjas: tuple[Franja, ...] = TODAS_LAS_FRANJAS


def _fechas_en_rango(inicio: date, fin: date):
    dias = (fin - inicio).days
    for i in range(dias + 1):
        yield inicio + timedelta(days=i)


class RankingDistritos:
    def __init__(
        self,
        predictor: PredictorFrecuenciaPort,
        distritos: DistritoRepositoryPort,
        clima: ClimaRepositoryPort,
        calendario_repo: CalendarioRepositoryPort,
        exposicion: ExposicionRepositoryPort,
        parque: ParqueRepositoryPort,
        terciles: TercilRepositoryPort,
    ) -> None:
        self._predictor = predictor
        self._distritos = distritos
        self._clima = clima
        self._calendario = calendario_repo
        self._exposicion = exposicion
        self._parque = parque
        self._terciles = terciles

    def _features_de_distrito(
        self, distrito: Distrito, fechas: list[date], franjas: tuple[Franja, ...]
    ) -> list[FeaturesCelda]:
        filas: list[FeaturesCelda] = []
        for fecha in fechas:
            prcp_mensual = self._clima.normal_climatologica(distrito.codigo, fecha.month)
            log_exposicion_v2 = self._exposicion.log_exposicion_v2(distrito.codigo, fecha)
            es_feriado = self._calendario.es_feriado(fecha, distrito.codigo)
            crec_parque_12m, _ = self._parque.crec_parque_12m(distrito.codigo, fecha)
            # Una fila por franja: franja_cod hace que el modelo prediga
            # distinto en cada una (el resto de las variables es de distrito+fecha).
            for franja in franjas:
                filas.append(
                    FeaturesCelda(
                        es_lluviosa=calendario.es_lluviosa(fecha),
                        es_finde=calendario.es_finde(fecha),
                        dia_semana=calendario.dia_semana(fecha),
                        periodo_agostino=calendario.periodo_agostino(fecha),
                        es_feriado=int(es_feriado),
                        prcp_mensual=prcp_mensual,
                        pct_urbano=distrito.pct_urbano,
                        poblacion_baja=int(distrito.poblacion_baja),
                        franja_cod=franja.codigo,
                        crec_parque_12m=crec_parque_12m,
                        pct_hogares_auto=distrito.pct_hogares_auto,
                        pct_trabaja_otro_lugar=distrito.pct_trabaja_otro_lugar,
                        log_exposicion_v2=log_exposicion_v2,
                    )
                )
        return filas

    def ejecutar(self, peticion: PeticionRanking) -> list[RankingDistrito]:
        fechas = list(_fechas_en_rango(peticion.fecha_inicio, peticion.fecha_fin))
        resultados: list[RankingDistrito] = []

        for distrito in self._distritos.listar():
            filas = self._features_de_distrito(distrito, fechas, peticion.franjas)
            predicciones = self._predictor.predecir_lote(filas)
            total = sum(predicciones)
            tercil, confiabilidad = self._terciles.tercil_y_confiabilidad(distrito.codigo)
            resultados.append(
                RankingDistrito(
                    distrito_codigo=distrito.codigo,
                    siniestros_esperados=total,
                    tercil=tercil,
                    confiabilidad_spearman=confiabilidad,
                )
            )

        resultados.sort(key=lambda r: r.siniestros_esperados, reverse=True)
        return resultados
