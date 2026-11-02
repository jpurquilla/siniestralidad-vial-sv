"""Modelos de dominio del prototipo de predicción de siniestralidad vial.

Python puro: cero imports de FastAPI, pandas, numpy o xgboost. Las decisiones
analíticas que fijan estos campos vienen de NB03/NB05/NB12 (ver api/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Franja(str, Enum):
    """Las cuatro franjas horarias del proyecto. El valor coincide carácter
    por carácter con la columna 'franja_horaria' de matriz_frecuencia_v3.csv.gz
    (NB03) y con las claves de 'codificacion_franja' de nb12_especificacion.json.
    El orden de definición es el de franja_cod (0-3); el adaptador del modelo
    lo valida contra la especificación al arrancar."""

    MADRUGADA = "Madrugada (00-05)"
    MANANA = "Manana (06-11)"
    TARDE = "Tarde (12-17)"
    NOCHE = "Noche (18-23)"

    @property
    def codigo(self) -> int:
        """Código numérico 0-3 (franja_cod) que consume el modelo."""
        return tuple(Franja).index(self)


class Tercil(str, Enum):
    ALTO = "alto"
    MEDIO = "medio"
    BAJO = "bajo"


class FuentePrecipitacion(str, Enum):
    OBSERVADA = "observada"
    NORMAL_CLIMATOLOGICA = "normal_climatologica"


class FuenteParque(str, Enum):
    """Origen de crec_parque_12m en una predicción: el valor observado del
    mes consultado (dentro de 2022-01..2026-06) o el del último mes
    disponible (fechas fuera de ese rango)."""

    OBSERVADO = "observado"
    ULTIMO_DISPONIBLE = "ultimo_disponible"


@dataclass(frozen=True)
class Distrito:
    codigo: str  # distrito_key normalizado (src.config.reconciliar_distrito)
    nombre: str
    departamento: str
    pct_urbano: float
    poblacion: int
    poblacion_baja: bool
    pct_hogares_auto: float  # censo 2024 (TAB_HOG_4), constante por distrito
    pct_trabaja_otro_lugar: float  # censo 2024 (TAB_CE_5), constante por distrito


# Orden normativo de las columnas del modelo final (NB12, 'ampliado'). El
# adaptador del modelo lo valida contra nb12_especificacion.json al arrancar.
COLUMNAS_MODELO: tuple[str, ...] = (
    "es_lluviosa",
    "es_finde",
    "dia_semana",
    "periodo_agostino",
    "es_feriado",
    "prcp_mensual",
    "pct_urbano",
    "poblacion_baja",
    "franja_cod",
    "crec_parque_12m",
    "pct_hogares_auto",
    "pct_trabaja_otro_lugar",
)


@dataclass(frozen=True)
class FeaturesCelda:
    """Las 12 variables predictoras del modelo final (NB12, 'ampliado'), en
    el orden normativo de nb12_especificacion.json, más log_exposicion_v2
    (NB03 base + tendencia NB05).

    A propósito NO incluye log_tasa_base: ese valor es un detalle del
    artefacto del modelo (vive en nb12_especificacion.json), no un concepto de
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
    franja_cod: int
    crec_parque_12m: float
    pct_hogares_auto: float
    pct_trabaja_otro_lugar: float
    log_exposicion_v2: float

    def como_lista(self) -> list[float]:
        """Las 12 columnas del modelo, en el orden de COLUMNAS_MODELO
        (log_exposicion_v2 NO va aquí: es offset)."""
        return [getattr(self, columna) for columna in COLUMNAS_MODELO]


@dataclass(frozen=True)
class PrediccionCelda:
    distrito_codigo: str
    fecha: date
    franja: Franja
    siniestros_esperados: float
    fuente_prcp: FuentePrecipitacion
    fuente_parque: FuenteParque
    prcp_mensual: float


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


@dataclass(frozen=True)
class ImpactoAnual:
    """Una fila de la serie oficial 2022-2026 (NB06). Real: los totales
    anuales son el dato oficial ONASEVI/FONAT contra el que se calibró el
    conjunto sintético (ver NB01) — no son sintéticos ellos mismos."""

    anio: str
    siniestros: int
    fallecidos: int
    letalidad: float
    cobertura: str


@dataclass(frozen=True)
class ImpactoResumenItem:
    """Una fila de `resumen` en nb06_resultados.json: variación interanual de
    una métrica en un período, con su lectura declarada (nunca causal —
    NB06 nunca afirma que la ley causó el cambio, solo lo describe)."""

    metrica: str
    periodo: str
    variacion_interanual_pct: float | None
    lectura: str


@dataclass(frozen=True)
class RepartoTipoUsuario:
    """Retrospectivo (NB09 §4.2): nunca predictivo. `grupo_vulnerable` es el
    más vulnerable presente en el siniestro, por prioridad fija peatón >
    ciclista > motociclista > otro."""

    tipo: str
    pct_fatales: float
    letalidad_por_siniestro: float


@dataclass(frozen=True)
class LetalidadPorEdad:
    """Retrospectivo (NB09 §4.1): rango_etario del conductor, o de la
    primera víctima si no hay conductor — ninguno consulta el desenlace."""

    rango_etario: str
    victimas: int
    letalidad: float


@dataclass(frozen=True)
class ComparacionDistrito:
    """Observado (n_siniestros de matriz_frecuencia_v3.csv.gz, conjunto
    sintético) contra predicho por el modelo final de NB12
    (XGBoostFrecuenciaAdapter), sumados sobre las filas de prueba de ese
    distrito."""

    distrito_codigo: str
    observado: float
    predicho: float

    @property
    def brecha_pct(self) -> float:
        if self.observado == 0:
            return 0.0
        return (1 - self.predicho / self.observado) * 100


@dataclass(frozen=True)
class SiniestroPunto:
    """Un siniestro individual real (sintético, calibrado — ver NB01), con
    coordenadas GPS reales de `siniestros.csv`. Para el mapa de hotspots
    (clusters de marcadores, replicando observatoriovial.fonat.gob.sv) —
    NB10 §8 encontró que estos puntos no tienen estructura espacial real por
    debajo de ~1 km: los clusters reflejan reparto por exposición, no
    concentraciones viales verificadas."""

    id_siniestro: str
    fecha: date
    hora: str
    franja: Franja
    latitud: float
    longitud: float
    departamento: str
    distrito: str
    tipo_siniestro: str
    factor_causante: str
    tipo_vehiculo: str
    fallecidos: int
    lesionados: int


@dataclass(frozen=True)
class FiltroSiniestros:
    fecha_inicio: date | None = None
    fecha_fin: date | None = None
    tipo_siniestro: str | None = None
    tipo_vehiculo: str | None = None
    factor_causante: str | None = None
    hora_inicio: str | None = None  # "HH:MM"
    hora_fin: str | None = None  # "HH:MM"
    departamento: str | None = None
    distrito: str | None = None


@dataclass(frozen=True)
class OpcionesFiltroSiniestros:
    tipos_siniestro: list[str]
    tipos_vehiculo: list[str]
    factores_causantes: list[str]
    departamentos: list[str]


@dataclass(frozen=True)
class GrupoEquidad:
    """Calibración (sum(predicho)/sum(observado); 1.0 = perfecta, >1 =
    sobreestima) y Spearman de orden predicho-vs-observado dentro de un grupo
    de distritos — misma definición de calibración que NB12 y misma
    metodología de corte que NB08 §4 (umbral_pct_urbano=0.3), aplicada acá
    también al corte por `poblacion_baja`."""

    grupo: str
    calibracion: float
    spearman: float
    n_distritos: int


@dataclass(frozen=True)
class TasaPorVehiculo:
    """Siniestralidad relativa al parque vehicular REAL (ONASEVI), por año.
    El parque es dato real; los siniestros/fallecidos son la serie oficial
    (NB06). Para el año parcial se usa el último corte dentro de la cobertura
    de la serie."""

    anio: str
    parque_vehicular: int
    fecha_corte_parque: date
    siniestros_por_1000_vehiculos: float
    fallecidos_por_10000_vehiculos: float
    parcial: bool
