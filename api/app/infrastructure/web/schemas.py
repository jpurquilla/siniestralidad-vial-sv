"""Schemas pydantic de request/response. Cero lógica de negocio: solo forma
y validación de entrada."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings
from app.domain.models import Franja, FuentePrecipitacion, Tercil

AVISO_DATOS_SINTETICOS = (
    "El conjunto de datos es sintético, calibrado sobre agregados de "
    "ONASEVI/FONAT (ver NB01). No usar para decisiones operativas sin "
    "validación con registros reales."
)

_FECHA_MINIMA = date.fromisoformat(settings.fecha_minima)
_FECHA_MAXIMA = date.fromisoformat(settings.fecha_maxima)


def _validar_fecha_en_rango(fecha: date) -> date:
    if not (_FECHA_MINIMA <= fecha <= _FECHA_MAXIMA):
        raise ValueError(
            f"La fecha {fecha.isoformat()} está fuera del rango admitido "
            f"[{_FECHA_MINIMA.isoformat()}, {_FECHA_MAXIMA.isoformat()}]."
        )
    return fecha


class DistritoOut(BaseModel):
    codigo: str
    nombre: str
    departamento: str
    pct_urbano: float
    poblacion: int
    poblacion_baja: bool


class DistritosResponse(BaseModel):
    total: int
    distritos: list[DistritoOut]


class PrediccionRequest(BaseModel):
    distritos: list[str] = Field(min_length=1, description="Códigos de distrito (ver /api/v1/distritos)")
    fecha_inicio: date
    fecha_fin: date
    franjas: list[Franja] = Field(default_factory=lambda: list(Franja))
    prcp_mensual: float | None = Field(
        default=None,
        ge=0,
        description="mm de precipitación mensual para simular un escenario. "
        "Si se omite, se usa la normal climatológica del distrito y mes.",
    )

    @field_validator("fecha_inicio", "fecha_fin")
    @classmethod
    def _fecha_en_rango(cls, fecha: date) -> date:
        return _validar_fecha_en_rango(fecha)

    @model_validator(mode="after")
    def _rango_valido(self) -> "PrediccionRequest":
        if self.fecha_fin < self.fecha_inicio:
            raise ValueError("fecha_fin no puede ser anterior a fecha_inicio.")
        dias = (self.fecha_fin - self.fecha_inicio).days + 1
        if dias > settings.max_dias_por_rango:
            raise ValueError(
                f"El rango solicitado tiene {dias} días; el máximo admitido es "
                f"{settings.max_dias_por_rango}."
            )
        if not self.franjas:
            raise ValueError("Debe especificarse al menos una franja horaria.")
        return self


class PrediccionCeldaOut(BaseModel):
    distrito_codigo: str
    fecha: date
    franja: Franja
    siniestros_esperados: float
    fuente_prcp: FuentePrecipitacion


class PrediccionResponse(BaseModel):
    aviso: str = AVISO_DATOS_SINTETICOS
    predicciones: list[PrediccionCeldaOut]


class RankingItemOut(BaseModel):
    distrito_codigo: str
    nombre: str
    siniestros_esperados: float
    tercil: Tercil
    confiabilidad_spearman: float = Field(
        description="Spearman de ordenamiento de ESE tercil en la evaluación de NB07 "
        "(test 2025-2026). El tercil medio ordena mal (~0.65): es una limitación "
        "documentada del modelo, no un defecto de este endpoint."
    )


class RankingResponse(BaseModel):
    aviso: str = AVISO_DATOS_SINTETICOS
    fecha_inicio: date
    fecha_fin: date
    ranking: list[RankingItemOut]


class RankingQuery(BaseModel):
    fecha_inicio: date
    fecha_fin: date
    franjas: list[Franja] = Field(default_factory=lambda: list(Franja))

    @field_validator("fecha_inicio", "fecha_fin")
    @classmethod
    def _fecha_en_rango(cls, fecha: date) -> date:
        return _validar_fecha_en_rango(fecha)

    @model_validator(mode="after")
    def _rango_valido(self) -> "RankingQuery":
        if self.fecha_fin < self.fecha_inicio:
            raise ValueError("fecha_fin no puede ser anterior a fecha_inicio.")
        dias = (self.fecha_fin - self.fecha_inicio).days + 1
        if dias > settings.max_dias_por_rango:
            raise ValueError(
                f"El rango solicitado tiene {dias} días; el máximo admitido es "
                f"{settings.max_dias_por_rango}."
            )
        return self


class ModeloInfoResponse(BaseModel):
    nombre: str
    fecha_entrenamiento: str
    variables: list[str]
    offset: str
    metricas: dict
    limitaciones: list[str]


class HealthResponse(BaseModel):
    estado: str
    modelo_cargado: bool
    log_tasa_base: float | None
    prediccion_canaria_ok: bool
    version: str


AVISO_RETROSPECTIVO = (
    "Esta pantalla describe el histórico 2022-2026. No predice nada y no debe "
    "usarse como estimación de riesgo futuro — ver /api/v1/modelo/info para lo "
    "que el modelo predictivo sí admite."
)

AVISO_COMPARACION = (
    "Comparación diagnóstica entre lo observado (conjunto sintético, calibrado "
    "sobre agregados de ONASEVI/FONAT — ver NB01) y lo predicho por el modelo de "
    "NB07 sobre esas mismas filas históricas. No es una predicción para fechas "
    "futuras."
)


class ImpactoAnualOut(BaseModel):
    anio: str
    siniestros: int
    fallecidos: int
    letalidad: float = Field(description="Fallecidos por cada 100 siniestros.")
    cobertura: str


class ImpactoResumenItemOut(BaseModel):
    metrica: str
    periodo: str
    variacion_interanual_pct: float | None
    lectura: str = Field(
        description="Nunca afirma causalidad: la ley coincide con el cambio, no lo causa."
    )


class ImpactoLeyResponse(BaseModel):
    aviso: str = AVISO_RETROSPECTIVO
    fecha_aprobacion_ley: date
    serie_anual: list[ImpactoAnualOut]
    resumen: list[ImpactoResumenItemOut]
    limitaciones: list[str]


class RepartoTipoUsuarioOut(BaseModel):
    tipo: str
    pct_fatales: float = Field(description="% de siniestros fatales de este tipo, sobre el total con víctima.")
    letalidad_por_siniestro: float = Field(
        description="Fracción de los siniestros de este tipo que tuvieron al menos un fallecido."
    )


class LetalidadPorEdadOut(BaseModel):
    rango_etario: str
    victimas: int
    letalidad: float


class UsuariosVulnerablesResponse(BaseModel):
    aviso: str = AVISO_RETROSPECTIVO
    reparto_por_tipo: list[RepartoTipoUsuarioOut]
    letalidad_por_edad: list[LetalidadPorEdadOut]


class ComparacionDistritoOut(BaseModel):
    distrito_codigo: str
    nombre: str
    observado: float
    predicho: float
    brecha_pct: float = Field(description="(1 - predicho/observado) * 100. Positivo = el modelo subestima.")


class ComparacionHistoricaResponse(BaseModel):
    aviso: str = AVISO_COMPARACION
    distritos: list[ComparacionDistritoOut]


class GrupoEquidadOut(BaseModel):
    grupo: str
    calibracion: float = Field(description="sum(observado)/sum(predicho) del grupo. 1.0 = calibración perfecta.")
    spearman: float = Field(description="Correlación de Spearman entre predicho y observado dentro del grupo.")
    n_distritos: int


class EquidadResponse(BaseModel):
    aviso: str = AVISO_COMPARACION
    urbano_rural: list[GrupoEquidadOut]
    densidad_poblacional: list[GrupoEquidadOut]


AVISO_HOTSPOTS = (
    "Estos puntos son sintéticos, calibrados sobre agregados de ONASEVI/FONAT "
    "(ver NB01). NB10 no encontró estructura espacial real por debajo de ~1 km "
    "en este conjunto (efecto medido: 0.71%, muy por debajo del umbral de "
    "significancia usado): los clusters reflejan el reparto por exposición "
    "poblacional, no concentraciones viales verificadas."
)


class SiniestroPuntoOut(BaseModel):
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


class SiniestrosPuntosResponse(BaseModel):
    aviso: str = AVISO_HOTSPOTS
    total: int
    puntos: list[SiniestroPuntoOut]


class OpcionesFiltroSiniestrosResponse(BaseModel):
    tipos_siniestro: list[str]
    tipos_vehiculo: list[str]
    factores_causantes: list[str]
    departamentos: list[str]
