"""Caso de uso: informe de siniestros esperados por periodo.

Agrega las predicciones que el modelo ya hace (los esperados son sumables) para un rango
de fechas, con filtros opcionales, y las descompone por causa según la proporción histórica
de cada causa por franja. Una sola llamada vectorizada al predictor (vía `PredecirCelda`,
para que coincida con la tarjeta del mapa de riesgo).

Python puro: orquesta puertos y casos de uso, sin importar FastAPI, pandas ni los
generadores de PDF/Excel.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.application.calcular_equidad import CalcularEquidad
from app.application.predecir_celda import (
    DistritoNoEncontradoError,
    PeticionPrediccionCelda,
    PredecirCelda,
)
from app.domain import calendario, poisson
from app.application.redactar_resumen import redactar_resumen
from app.domain.colores_causas import (
    EBRIEDAD,
    NUM_CAUSAS_PRINCIPALES,
    OTRAS_CAUSAS,
    colores_para,
    nombre_causa,
)
from app.domain.informe import (
    DIAS_SEMANA,
    CausasDeFranja,
    ComposicionVehiculos,
    FilaCausa,
    FilaDiaSemana,
    FilaDistrito,
    FilaDistritoFranja,
    FilaFranja,
    FiltrosInforme,
    InformePeriodo,
    ParametrosInforme,
    Prioridad,
    SupuestosInforme,
    TotalInforme,
)
from app.domain.models import ComposicionCausas, Distrito, Franja, FuenteParque, ProporcionCausa
from app.domain.models import Tercil
from app.domain.ports import DistritoRepositoryPort

MAX_DIAS_INFORME = 92
NUM_PRIORIDADES = 10
TASA_POR = 10_000


class PeticionInformeInvalidaError(ValueError):
    """Parámetros del informe inválidos (el router los traduce a 422)."""


@dataclass(frozen=True)
class PeticionInformePeriodo:
    fecha_inicio: date
    fecha_fin: date
    departamento: str | None = None
    distritos: tuple[str, ...] = ()
    franjas: tuple[Franja, ...] = ()
    dias_semana: tuple[int, ...] = ()  # 0 = lunes


def _sin_repetidos(valores):
    return tuple(dict.fromkeys(valores))


def repartir_terciles(n: int) -> list[Tercil]:
    """Tercil de cada posición 0..n-1 de un ranking descendente: tres grupos de tamaño ⌈n/3⌉/⌊n/3⌋
    (103 -> 35, 34, 34; el resto se reparte primero a los grupos más altos; n=1 -> alto)."""
    base, resto = divmod(n, 3)
    tamanos = [base + (1 if i < resto else 0) for i in range(3)]
    terciles: list[Tercil] = []
    for tercil, tamano in zip((Tercil.ALTO, Tercil.MEDIO, Tercil.BAJO), tamanos, strict=True):
        terciles.extend([tercil] * tamano)
    return terciles


def _fechas(inicio: date, fin: date) -> list[date]:
    return [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]


class GenerarInformePeriodo:
    def __init__(
        self,
        predecir_celda: PredecirCelda,
        distritos: DistritoRepositoryPort,
        composicion_causas: ComposicionCausas,
        composicion_vehiculos: ComposicionVehiculos,
        calcular_equidad: CalcularEquidad,
        reloj: Callable[[], datetime],
    ) -> None:
        self._predecir = predecir_celda
        self._distritos = distritos
        self._composicion = composicion_causas
        self._vehiculos = composicion_vehiculos
        self._equidad = calcular_equidad
        self._reloj = reloj  # devuelve un datetime con zona horaria (inyectable para tests)

    # ------------------------------------------------------------------ validación

    def _validar(self, p: PeticionInformePeriodo) -> PeticionInformePeriodo:
        if p.fecha_fin < p.fecha_inicio:
            raise PeticionInformeInvalidaError("fecha_fin no puede ser anterior a fecha_inicio.")
        dias = (p.fecha_fin - p.fecha_inicio).days + 1
        if dias > MAX_DIAS_INFORME:
            raise PeticionInformeInvalidaError(
                f"El rango solicitado tiene {dias} días; el máximo admitido para el informe "
                f"es {MAX_DIAS_INFORME}."
            )
        if any(not 0 <= d <= 6 for d in p.dias_semana):
            raise PeticionInformeInvalidaError(
                "dias_semana debe contener enteros de 0 (lunes) a 6 (domingo)."
            )
        return PeticionInformePeriodo(
            fecha_inicio=p.fecha_inicio,
            fecha_fin=p.fecha_fin,
            departamento=p.departamento or None,
            distritos=_sin_repetidos(p.distritos),
            franjas=_sin_repetidos(p.franjas),
            dias_semana=_sin_repetidos(p.dias_semana),
        )

    def _distritos_del_filtro(self, p: PeticionInformePeriodo) -> list[Distrito]:
        if p.distritos:
            seleccion: list[Distrito] = []
            for codigo in p.distritos:
                distrito = self._distritos.obtener(codigo)
                if distrito is None:
                    raise DistritoNoEncontradoError(codigo)
                seleccion.append(distrito)
        else:
            seleccion = self._distritos.listar()

        if p.departamento:
            departamento = p.departamento.strip().casefold()
            conocidos = {d.departamento for d in self._distritos.listar()}
            if departamento not in {c.casefold() for c in conocidos}:
                raise PeticionInformeInvalidaError(
                    f"Departamento desconocido: {p.departamento!r}. "
                    f"Valores válidos: {', '.join(sorted(conocidos))}."
                )
            seleccion = [d for d in seleccion if d.departamento.casefold() == departamento]
        if not seleccion:
            raise PeticionInformeInvalidaError(
                "Los filtros de departamento y distritos no coinciden con ningún distrito."
            )
        return seleccion

    # ------------------------------------------------------------------ caso de uso

    def ejecutar(self, solicitud: PeticionInformePeriodo) -> InformePeriodo:
        p = self._validar(solicitud)
        seleccion = self._distritos_del_filtro(p)
        franjas = [f for f in Franja if f in p.franjas] if p.franjas else list(Franja)
        fechas = [
            f for f in _fechas(p.fecha_inicio, p.fecha_fin)
            if not p.dias_semana or calendario.dia_semana(f) in p.dias_semana
        ]
        if not fechas:
            raise PeticionInformeInvalidaError(
                "Ningún día del rango coincide con los días de la semana elegidos."
            )

        celdas = [(d, f, fr) for d in seleccion for f in fechas for fr in franjas]
        resultados = self._predecir.esperados_lote(
            [PeticionPrediccionCelda(d.codigo, f, fr) for d, f, fr in celdas]
        )

        por_distrito_franja: dict[tuple[str, Franja], float] = defaultdict(float)
        por_dia: list[float] = [0.0] * 7
        veces_dia = [0] * 7
        for f in fechas:
            veces_dia[calendario.dia_semana(f)] += 1
        por_prioridad: dict[tuple[str, Franja, int], float] = defaultdict(float)
        fechas_con_ultimo_parque: set[date] = set()
        for (distrito, fecha, franja), (esperado, fuente_parque) in zip(celdas, resultados, strict=True):
            dow = calendario.dia_semana(fecha)
            por_distrito_franja[(distrito.codigo, franja)] += esperado
            por_dia[dow] += esperado
            por_prioridad[(distrito.codigo, franja, dow)] += esperado
            if fuente_parque is FuenteParque.ULTIMO_DISPONIBLE:
                fechas_con_ultimo_parque.add(fecha)

        total_distrito = {
            d.codigo: sum(por_distrito_franja[(d.codigo, fr)] for fr in franjas) for d in seleccion
        }
        total_franja = {fr: sum(por_distrito_franja[(d.codigo, fr)] for d in seleccion) for fr in franjas}
        total = sum(total_distrito.values())

        nombre_de = {d.codigo: d.nombre for d in seleccion}
        ordenados = sorted(seleccion, key=lambda d: (-total_distrito[d.codigo], d.codigo))

        # El tercil se recalcula con el propio informe: el orden de la tabla y el tercil coinciden.
        terciles = repartir_terciles(len(ordenados))
        por_distrito = tuple(
            self._fila_distrito(d, total_distrito[d.codigo], tercil)
            for d, tercil in zip(ordenados, terciles, strict=True)
        )
        por_causa, principales = self._causas(total_franja, total, franjas)
        por_dia_semana = tuple(
            FilaDiaSemana(
                DIAS_SEMANA[i],
                por_dia[i] / veces_dia[i] if veces_dia[i] else 0.0,
                veces_dia[i],
                por_dia[i],
            )
            for i in range(7)
        )
        causas_por_franja = tuple(
            CausasDeFranja(fr, self._proporciones_agrupadas(fr, principales)) for fr in franjas
        )
        resumen = redactar_resumen(
            fecha_inicio=p.fecha_inicio,
            fecha_fin=p.fecha_fin,
            total=total,
            ic90=poisson.intervalo_90(total),
            distritos=[(d.nombre, total_distrito[d.codigo]) for d in ordenados],
            por_franja=[(fr, total_franja[fr]) for fr in franjas],
            por_dia=por_dia_semana,
            ebriedad_por_franja={
                fr: next((x.proporcion for x in self._proporciones_por_nombre(fr) if x.causa == EBRIEDAD), 0.0)
                for fr in franjas
            },
        )

        return InformePeriodo(
            parametros=ParametrosInforme(
                fecha_inicio=p.fecha_inicio,
                fecha_fin=p.fecha_fin,
                filtros=FiltrosInforme(p.departamento, p.distritos, p.franjas, p.dias_semana),
            ),
            generado=self._reloj(),
            total=TotalInforme(total, *poisson.intervalo_90(total)),
            por_distrito=por_distrito,
            por_franja=tuple(FilaFranja(fr, total_franja.get(fr, 0.0)) for fr in Franja),
            por_dia_semana=por_dia_semana,
            distrito_x_franja=tuple(
                FilaDistritoFranja(
                    d.codigo, d.nombre, {fr: por_distrito_franja.get((d.codigo, fr), 0.0) for fr in Franja}
                )
                for d in ordenados
            ),
            por_causa=por_causa,
            causas_por_franja=causas_por_franja,
            prioridades=tuple(
                Prioridad(nombre_de[codigo], franja, DIAS_SEMANA[dow], suma / veces_dia[dow], veces_dia[dow])
                for (codigo, franja, dow), suma in sorted(
                    por_prioridad.items(),
                    # Promedio por día desc; desempate estable: mayor suma, nombre, franja, día.
                    key=lambda kv: (
                        -round(kv[1] / veces_dia[kv[0][2]], 9), -kv[1],
                        nombre_de[kv[0][0]], kv[0][1].codigo, kv[0][2],
                    ),
                )[:NUM_PRIORIDADES]
            ),
            colores_causas=colores_para(
                [nombre_causa(c.causa) for pf in self._composicion.por_franja.values() for c in pf]
            ),
            supuestos=SupuestosInforme(
                dias_con_lluvia_normal=len(fechas),  # el informe nunca recibe lluvia observada
                dias_con_ultimo_parque=len(fechas_con_ultimo_parque),
                composicion_vehiculos=self._vehiculos.porcentajes,
            ),
            limitaciones=self._limitaciones(),
            resumen=resumen,
        )

    # ------------------------------------------------------------------ piezas

    def _proporciones_por_nombre(self, franja: Franja) -> tuple[ProporcionCausa, ...]:
        """Proporciones históricas de la franja con el nombre de causa ya presentable."""
        return tuple(ProporcionCausa(nombre_causa(p.causa), p.proporcion)
                     for p in self._composicion.proporciones(franja))

    def _fila_distrito(self, d: Distrito, esperado: float, tercil: Tercil) -> FilaDistrito:
        inf, sup = poisson.intervalo_90(esperado)
        return FilaDistrito(
            codigo=d.codigo,
            distrito=d.nombre,
            departamento=d.departamento,
            esperado=esperado,
            ic90_inf=inf,
            ic90_sup=sup,
            tercil=tercil,
            tasa_10mil_hab=esperado / d.poblacion * TASA_POR if d.poblacion else 0.0,
        )

    def _causas(
        self, total_franja: dict[Franja, float], total: float, franjas: list[Franja]
    ) -> tuple[tuple[FilaCausa, ...], tuple[str, ...]]:
        """Esperado por causa = Σ_franja esperado_de_la_franja × proporción de la causa en la
        franja. Devuelve las filas (8 principales + «Otras causas») y los nombres principales."""
        por_causa: dict[str, float] = defaultdict(float)
        for franja in franjas:
            for prop in self._proporciones_por_nombre(franja):
                por_causa[prop.causa] += total_franja[franja] * prop.proporcion

        ranking = sorted(por_causa.items(), key=lambda kv: (-kv[1], kv[0]))
        principales = ranking[:NUM_CAUSAS_PRINCIPALES]
        otras = sum(v for _, v in ranking[NUM_CAUSAS_PRINCIPALES:])
        filas = [*principales, (OTRAS_CAUSAS, otras)]
        return (
            tuple(
                FilaCausa(causa, esperado, *poisson.intervalo_90(esperado),
                          porcentaje=esperado / total * 100 if total > 0 else 0.0)
                for causa, esperado in filas
            ),
            tuple(c for c, _ in principales),
        )

    def _proporciones_agrupadas(
        self, franja: Franja, principales: tuple[str, ...]
    ) -> tuple[ProporcionCausa, ...]:
        proporcion = {p.causa: p.proporcion for p in self._proporciones_por_nombre(franja)}
        resto = sum(v for c, v in proporcion.items() if c not in principales)
        return (
            *(ProporcionCausa(c, proporcion.get(c, 0.0)) for c in principales),
            ProporcionCausa(OTRAS_CAUSAS, resto),
        )

    def _limitaciones(self) -> tuple[str, ...]:
        rural = next(g for g in self._equidad.ejecutar().urbano_rural if g.grupo == "Rural")
        desvio = round(abs(rural.calibracion - 1) * 100)
        verbo = "sobreestimó" if rural.calibracion >= 1 else "subestimó"
        porcentajes = ", ".join(f"{v.tipo} {v.porcentaje:.1f}%" for v in self._vehiculos.porcentajes)
        return (
            "Datos sintéticos, calibrados sobre agregados de ONASEVI/FONAT; las reglas con que "
            "se generaron no están documentadas.",
            "Para fechas futuras se usa la lluvia normal del mes y el último dato disponible de "
            "parque vehicular.",
            "Las causas son una composición promedio por franja horaria; no indican la causa de "
            "siniestros en particular. En el conjunto de datos la composición varía según la "
            "franja, pero no entre distritos ni días de la semana.",
            "En el conjunto de datos, el tipo de vehículo involucrado no varía de forma "
            f"apreciable entre distritos, franjas horarias ni causas ({porcentajes}), por "
            "lo que el informe no lo desglosa.",
            f"En el periodo de evaluación el modelo {verbo} un {desvio}% los siniestros en los "
            "distritos rurales (menos de 30% de población urbana).",
            "El rango del 90% refleja solo la variación aleatoria (Poisson) dada la predicción; "
            "no incluye el error del propio modelo.",
        )
