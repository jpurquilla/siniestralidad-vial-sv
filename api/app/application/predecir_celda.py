"""Caso de uso: predecir siniestros esperados para una celda distrito×fecha×franja.

Python puro: orquesta los puertos del dominio, sin importar FastAPI, pandas
ni xgboost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.domain import calendario
from app.domain.models import (
    ComposicionCausas,
    FeaturesCelda,
    Franja,
    FuenteParque,
    FuentePrecipitacion,
    PrediccionCelda,
)
from app.domain.ports import (
    CalendarioRepositoryPort,
    ClimaRepositoryPort,
    DistritoRepositoryPort,
    ExposicionRepositoryPort,
    ParqueRepositoryPort,
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
    # Escenario relativo: prcp = normal del distrito × (1 + factor_prcp), limitada
    # al rango observado en el entrenamiento. Excluyente con prcp_mensual.
    factor_prcp: float | None = None


class PredecirCelda:
    def __init__(
        self,
        predictor: PredictorFrecuenciaPort,
        distritos: DistritoRepositoryPort,
        clima: ClimaRepositoryPort,
        calendario_repo: CalendarioRepositoryPort,
        exposicion: ExposicionRepositoryPort,
        parque: ParqueRepositoryPort,
        composicion_causas: ComposicionCausas,
    ) -> None:
        self._predictor = predictor
        self._distritos = distritos
        self._clima = clima
        self._calendario = calendario_repo
        self._exposicion = exposicion
        self._parque = parque
        self._composicion_causas = composicion_causas

    def construir_features(
        self, peticion: PeticionPrediccionCelda
    ) -> tuple[FeaturesCelda, FuentePrecipitacion, FuenteParque]:
        distrito = self._distritos.obtener(peticion.distrito_codigo)
        if distrito is None:
            raise DistritoNoEncontradoError(peticion.distrito_codigo)

        if peticion.prcp_mensual is not None and peticion.factor_prcp is not None:
            raise ValueError("prcp_mensual y factor_prcp son excluyentes.")

        if peticion.prcp_mensual is not None:
            prcp_mensual = peticion.prcp_mensual
            fuente_prcp = FuentePrecipitacion.OBSERVADA
        elif peticion.factor_prcp is not None:
            normal = self._clima.normal_climatologica(distrito.codigo, peticion.fecha.month)
            minimo, maximo = self._clima.rango_entrenamiento()
            prcp_mensual = min(maximo, max(minimo, normal * (1 + peticion.factor_prcp)))
            fuente_prcp = FuentePrecipitacion.ESCENARIO_RELATIVO
        else:
            prcp_mensual = self._clima.normal_climatologica(distrito.codigo, peticion.fecha.month)
            fuente_prcp = FuentePrecipitacion.NORMAL_CLIMATOLOGICA

        log_exposicion_v2 = self._exposicion.log_exposicion_v2(distrito.codigo, peticion.fecha)
        es_feriado = self._calendario.es_feriado(peticion.fecha, distrito.codigo)
        crec_parque_12m, fuente_parque = self._parque.crec_parque_12m(distrito.codigo, peticion.fecha)

        features = FeaturesCelda(
            es_lluviosa=calendario.es_lluviosa(peticion.fecha),
            es_finde=calendario.es_finde(peticion.fecha),
            dia_semana=calendario.dia_semana(peticion.fecha),
            periodo_agostino=calendario.periodo_agostino(peticion.fecha),
            es_feriado=int(es_feriado),
            prcp_mensual=prcp_mensual,
            pct_urbano=distrito.pct_urbano,
            poblacion_baja=int(distrito.poblacion_baja),
            franja_cod=peticion.franja.codigo,
            crec_parque_12m=crec_parque_12m,
            pct_hogares_auto=distrito.pct_hogares_auto,
            pct_trabaja_otro_lugar=distrito.pct_trabaja_otro_lugar,
            log_exposicion_v2=log_exposicion_v2,
        )
        return features, fuente_prcp, fuente_parque

    def ejecutar(self, peticion: PeticionPrediccionCelda) -> PrediccionCelda:
        return self.ejecutar_lote([peticion])[0]

    def esperados_lote(
        self, peticiones: list[PeticionPrediccionCelda]
    ) -> list[tuple[float, FuenteParque]]:
        """Solo el esperado y la fuente del parque de cada celda, con UNA llamada vectorizada
        al predictor. Mismos features y mismo resultado que `ejecutar_lote`, pero sin
        construir el desglose por causa de cada celda (el informe por periodo lo hace sobre
        los totales por franja, no celda a celda)."""
        construidas = [self.construir_features(p) for p in peticiones]
        esperados = self._predictor.predecir_lote([features for features, _, _ in construidas])
        return [
            (esperado, fuente_parque)
            for (_, _, fuente_parque), esperado in zip(construidas, esperados, strict=True)
        ]

    def ejecutar_lote(self, peticiones: list[PeticionPrediccionCelda]) -> list[PrediccionCelda]:
        """Predice todas las celdas con UNA sola llamada al predictor
        (vectorizada), en el mismo orden que `peticiones`."""
        construidas = [self.construir_features(p) for p in peticiones]
        esperados = self._predictor.predecir_lote([features for features, _, _ in construidas])
        return [
            PrediccionCelda(
                distrito_codigo=peticion.distrito_codigo,
                fecha=peticion.fecha,
                franja=peticion.franja,
                siniestros_esperados=esperado,
                fuente_prcp=fuente_prcp,
                fuente_parque=fuente_parque,
                prcp_mensual=features.prcp_mensual,
                por_causa=tuple(self._composicion_causas.descomponer(esperado, peticion.franja)),
            )
            for peticion, (features, fuente_prcp, fuente_parque), esperado in zip(
                peticiones, construidas, esperados, strict=True
            )
        ]
