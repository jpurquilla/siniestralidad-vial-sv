"""Schemas pydantic del informe por periodo (contrato JSON compartido con la web).
Cero lógica de negocio: forma y validación de entrada, y mapeo desde el dominio."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

from app.domain.informe import InformePeriodo
from app.domain.models import Franja, Tercil
from app.infrastructure.web.schemas import _validar_fecha_en_rango

_DECIMALES = 6


def _r(valor: float) -> float:
    return round(valor, _DECIMALES)


class InformePeriodoRequest(BaseModel):
    fecha_inicio: date
    fecha_fin: date
    departamento: str | None = Field(default=None, description="Nombre con tildes, como en /distritos.")
    distritos: list[str] = Field(default_factory=list, description="Códigos de distrito (ver /api/v1/distritos).")
    franjas: list[Franja] = Field(default_factory=list)
    dias_semana: list[int] = Field(default_factory=list, description="0 = lunes ... 6 = domingo.")

    @field_validator("fecha_inicio", "fecha_fin")
    @classmethod
    def _fecha_en_rango(cls, fecha: date) -> date:
        return _validar_fecha_en_rango(fecha)

    @field_validator("dias_semana")
    @classmethod
    def _dias_validos(cls, dias: list[int]) -> list[int]:
        if any(not 0 <= d <= 6 for d in dias):
            raise ValueError("dias_semana debe contener enteros de 0 (lunes) a 6 (domingo).")
        return dias


class FiltrosOut(BaseModel):
    departamento: str | None
    distritos: list[str]
    franjas: list[Franja]
    dias_semana: list[int]


class ParametrosOut(BaseModel):
    fecha_inicio: date
    fecha_fin: date
    filtros: FiltrosOut


class TotalOut(BaseModel):
    esperado: float
    ic90_inf: int
    ic90_sup: int


class DistritoInformeOut(BaseModel):
    codigo: str
    distrito: str
    departamento: str
    esperado: float
    ic90_inf: int
    ic90_sup: int
    tercil: Tercil
    tasa_10mil_hab: float


class FranjaInformeOut(BaseModel):
    franja: Franja
    esperado: float


class DiaSemanaInformeOut(BaseModel):
    dia: str
    esperado: float = Field(description="Promedio por día: suma / veces (0.0 si el filtro excluye el día).")
    veces: int = Field(description="Veces que ese día de la semana aparece en el periodo filtrado.")
    suma: float = Field(description="Total esperado de ese día de la semana en el periodo.")


class DistritoFranjaOut(BaseModel):
    codigo: str
    distrito: str
    franjas: dict[Franja, float]


class CausaInformeOut(BaseModel):
    causa: str
    esperado: float
    ic90_inf: int
    ic90_sup: int
    porcentaje: float = Field(description="0-100; las filas suman 100.")


class ProporcionInformeOut(BaseModel):
    causa: str
    proporcion: float


class CausasFranjaOut(BaseModel):
    franja: Franja
    causas: list[ProporcionInformeOut]


class PrioridadOut(BaseModel):
    distrito: str
    franja: Franja
    dia_semana: str
    esperado: float = Field(description="Promedio por día de la combinación (suma ÷ veces).")
    veces: int


class VehiculoOut(BaseModel):
    tipo: str
    porcentaje: float


class SupuestosOut(BaseModel):
    dias_con_lluvia_normal: int
    dias_con_ultimo_parque: int
    composicion_vehiculos: list[VehiculoOut]


class FilaResumenOut(BaseModel):
    etiqueta: str
    valor: str


class ResumenOut(BaseModel):
    texto: str = Field(description="Párrafo en lenguaje llano generado de los datos del informe.")
    filas: list[FilaResumenOut] = Field(description="Tabla de exactamente 4 filas.")


class InformePeriodoResponse(BaseModel):
    parametros: ParametrosOut
    generado: str = Field(description="ISO 8601 con offset, zona America/El_Salvador.")
    resumen: ResumenOut
    total: TotalOut
    por_distrito: list[DistritoInformeOut]
    por_franja: list[FranjaInformeOut]
    por_dia_semana: list[DiaSemanaInformeOut]
    distrito_x_franja: list[DistritoFranjaOut]
    por_causa: list[CausaInformeOut]
    causas_por_franja: list[CausasFranjaOut]
    prioridades: list[PrioridadOut]
    colores_causas: dict[str, str]
    supuestos: SupuestosOut
    limitaciones: list[str]

    @classmethod
    def desde_dominio(cls, i: InformePeriodo) -> "InformePeriodoResponse":
        f = i.parametros.filtros
        return cls(
            parametros=ParametrosOut(
                fecha_inicio=i.parametros.fecha_inicio,
                fecha_fin=i.parametros.fecha_fin,
                filtros=FiltrosOut(
                    departamento=f.departamento,
                    distritos=list(f.distritos),
                    franjas=list(f.franjas),
                    dias_semana=list(f.dias_semana),
                ),
            ),
            generado=i.generado.isoformat(timespec="seconds"),
            resumen=ResumenOut(
                texto=i.resumen.texto,
                filas=[FilaResumenOut(etiqueta=x.etiqueta, valor=x.valor) for x in i.resumen.filas],
            ),
            total=TotalOut(esperado=_r(i.total.esperado), ic90_inf=i.total.ic90_inf, ic90_sup=i.total.ic90_sup),
            por_distrito=[
                DistritoInformeOut(
                    codigo=d.codigo, distrito=d.distrito, departamento=d.departamento,
                    esperado=_r(d.esperado), ic90_inf=d.ic90_inf, ic90_sup=d.ic90_sup,
                    tercil=d.tercil, tasa_10mil_hab=_r(d.tasa_10mil_hab),
                )
                for d in i.por_distrito
            ],
            por_franja=[FranjaInformeOut(franja=x.franja, esperado=_r(x.esperado)) for x in i.por_franja],
            por_dia_semana=[DiaSemanaInformeOut(dia=x.dia, esperado=_r(x.esperado), veces=x.veces, suma=_r(x.suma)) for x in i.por_dia_semana],
            distrito_x_franja=[
                DistritoFranjaOut(
                    codigo=x.codigo, distrito=x.distrito,
                    franjas={fr: _r(v) for fr, v in x.franjas.items()},
                )
                for x in i.distrito_x_franja
            ],
            por_causa=[
                CausaInformeOut(
                    causa=c.causa, esperado=_r(c.esperado), ic90_inf=c.ic90_inf, ic90_sup=c.ic90_sup,
                    porcentaje=_r(c.porcentaje),
                )
                for c in i.por_causa
            ],
            causas_por_franja=[
                CausasFranjaOut(
                    franja=x.franja,
                    causas=[ProporcionInformeOut(causa=c.causa, proporcion=c.proporcion) for c in x.causas],
                )
                for x in i.causas_por_franja
            ],
            prioridades=[
                PrioridadOut(
                    distrito=x.distrito, franja=x.franja, dia_semana=x.dia_semana,
                    esperado=_r(x.esperado), veces=x.veces,
                )
                for x in i.prioridades
            ],
            colores_causas=dict(i.colores_causas),
            supuestos=SupuestosOut(
                dias_con_lluvia_normal=i.supuestos.dias_con_lluvia_normal,
                dias_con_ultimo_parque=i.supuestos.dias_con_ultimo_parque,
                composicion_vehiculos=[
                    VehiculoOut(tipo=v.tipo, porcentaje=_r(v.porcentaje)) for v in i.supuestos.composicion_vehiculos
                ],
            ),
            limitaciones=list(i.limitaciones),
        )
