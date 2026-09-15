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
