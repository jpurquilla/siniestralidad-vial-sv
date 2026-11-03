"""Modelos del informe de siniestros esperados por periodo. Python puro.

El informe es la agregación de las predicciones que el modelo ya hace (los esperados son
sumables): no hay modelo nuevo. La causa se conoce después del siniestro; el desglose por
causa reparte lo esperado según la proporción histórica de cada causa por franja.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from app.domain.models import Franja, ProporcionCausa, Tercil

DIAS_SEMANA: tuple[str, ...] = (
    "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo",
)  # índice = calendario.dia_semana (0 = lunes)


@dataclass(frozen=True)
class PorcentajeVehiculo:
    tipo: str  # con tilde de presentación (Automóvil, Camión…), como el filtro de hotspots
    porcentaje: float  # 0-100


@dataclass(frozen=True)
class ComposicionVehiculos:
    """Porcentaje de cada tipo de vehículo en siniestros.csv 2022-2024, de mayor a menor."""

    periodo_inicio: date
    periodo_fin: date
    porcentajes: tuple[PorcentajeVehiculo, ...]


@dataclass(frozen=True)
class FiltrosInforme:
    departamento: str | None
    distritos: tuple[str, ...]
    franjas: tuple[Franja, ...]
    dias_semana: tuple[int, ...]


@dataclass(frozen=True)
class ParametrosInforme:
    fecha_inicio: date
    fecha_fin: date
    filtros: FiltrosInforme


@dataclass(frozen=True)
class TotalInforme:
    esperado: float
    ic90_inf: int
    ic90_sup: int


@dataclass(frozen=True)
class FilaDistrito:
    codigo: str
    distrito: str
    departamento: str
    esperado: float
    ic90_inf: int
    ic90_sup: int
    tercil: Tercil
    tasa_10mil_hab: float


@dataclass(frozen=True)
class FilaFranja:
    franja: Franja
    esperado: float


@dataclass(frozen=True)
class FilaDiaSemana:
    dia: str
    esperado: float  # PROMEDIO por día: suma / veces (0.0 si el día no aparece)
    veces: int  # cuántas veces aparece ese día de la semana en el periodo filtrado
    suma: float  # total esperado de ese día de la semana en el periodo


@dataclass(frozen=True)
class FilaDistritoFranja:
    codigo: str
    distrito: str
    franjas: dict[Franja, float]


@dataclass(frozen=True)
class FilaCausa:
    causa: str
    esperado: float
    ic90_inf: int
    ic90_sup: int
    porcentaje: float  # 0-100


@dataclass(frozen=True)
class CausasDeFranja:
    franja: Franja
    causas: tuple[ProporcionCausa, ...]


@dataclass(frozen=True)
class Prioridad:
    distrito: str
    franja: Franja
    dia_semana: str
    esperado: float  # promedio por día de la combinación (suma ÷ veces)
    veces: int  # veces que ese día de la semana aparece en el periodo


@dataclass(frozen=True)
class SupuestosInforme:
    dias_con_lluvia_normal: int
    dias_con_ultimo_parque: int
    composicion_vehiculos: tuple[PorcentajeVehiculo, ...]


@dataclass(frozen=True)
class FilaResumen:
    etiqueta: str
    valor: str


@dataclass(frozen=True)
class ResumenInforme:
    """Texto llano y tabla de 4 filas generados con los mismos datos del informe (fuente única)."""

    texto: str
    filas: tuple[FilaResumen, ...]


@dataclass(frozen=True)
class InformePeriodo:
    parametros: ParametrosInforme
    generado: datetime
    total: TotalInforme
    por_distrito: tuple[FilaDistrito, ...]
    por_franja: tuple[FilaFranja, ...]
    por_dia_semana: tuple[FilaDiaSemana, ...]
    distrito_x_franja: tuple[FilaDistritoFranja, ...]
    por_causa: tuple[FilaCausa, ...]
    causas_por_franja: tuple[CausasDeFranja, ...]
    prioridades: tuple[Prioridad, ...]
    colores_causas: dict[str, str]
    supuestos: SupuestosInforme
    limitaciones: tuple[str, ...]
    resumen: ResumenInforme
