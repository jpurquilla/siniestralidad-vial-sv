"""Modelos de dominio del prototipo de predicción de siniestralidad vial.

Python puro: cero imports de FastAPI, pandas, numpy o xgboost. Las decisiones
analíticas que fijan estos campos vienen de NB03/NB05/NB07 (ver api/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Franja(str, Enum):
    """Las cuatro franjas horarias del proyecto. El valor coincide carácter
    por carácter con la columna 'franja_horaria' de matriz_frecuencia_v2.csv
    (NB03) — el adaptador del modelo depende de esa igualdad literal."""

    MADRUGADA = "Madrugada (00-05)"
    MANANA = "Manana (06-11)"
    TARDE = "Tarde (12-17)"
    NOCHE = "Noche (18-23)"


class Tercil(str, Enum):
    ALTO = "alto"
    MEDIO = "medio"
    BAJO = "bajo"


class FuentePrecipitacion(str, Enum):
    OBSERVADA = "observada"
    NORMAL_CLIMATOLOGICA = "normal_climatologica"


@dataclass(frozen=True)
class Distrito:
    codigo: str  # distrito_key normalizado (src.config.reconciliar_distrito)
    nombre: str
    departamento: str
    pct_urbano: float
    poblacion: int
    poblacion_baja: bool


@dataclass(frozen=True)
class FeaturesCelda:
    """Las 8 variables predictoras del modelo de frecuencia (NB07), en el
    orden normativo de nb07_especificacion.json, más log_exposicion_v2 (NB03
    base + tendencia NB05).

    A propósito NO incluye log_tasa_base: ese valor es un detalle del
    artefacto de NB07 (vive en nb07_especificacion.json), no un concepto de
    dominio. El adaptador del modelo (infrastructure/modelo/xgboost_adapter.py)
    es el único responsable de sumarlo al construir el base_margin real de
    XGBoost — así la capa de aplicación no puede, por diseño, olvidar
    sumarlo."""

    es_lluviosa: int
    es_finde: int
    dia_semana: int
    periodo_agostino: int
    es_feriado: int
    prcp_mensual: float
    pct_urbano: float
    poblacion_baja: int
    log_exposicion_v2: float

    def como_lista(self) -> list[float]:
        """Orden normativo: es_lluviosa, es_finde, dia_semana, periodo_agostino,
        es_feriado, prcp_mensual, pct_urbano, poblacion_baja."""
        return [
            self.es_lluviosa,
            self.es_finde,
            self.dia_semana,
            self.periodo_agostino,
            self.es_feriado,
            self.prcp_mensual,
            self.pct_urbano,
            self.poblacion_baja,
        ]


@dataclass(frozen=True)
class PrediccionCelda:
    distrito_codigo: str
    fecha: date
    franja: Franja
    siniestros_esperados: float
    fuente_prcp: FuentePrecipitacion


@dataclass(frozen=True)
class RankingDistrito:
    distrito_codigo: str
    siniestros_esperados: float
    tercil: Tercil
    confiabilidad_spearman: float


@dataclass(frozen=True)
class RangoConsulta:
    fecha_inicio: date
    fecha_fin: date
    distritos: list[str]
    franjas: list[Franja]
