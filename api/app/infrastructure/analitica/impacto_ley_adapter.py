"""Adaptador de impacto de la Ley Cero Tolerancia (NB06). Retrospectivo y
real: lee `nb06_resultados.json`, ya generado y verificado por NB06 — este
adaptador no recalcula nada, solo da forma al dato existente.

Se carga y valida una sola vez, en el lifespan (ver app/main.py), como el
resto de los adaptadores."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.domain.models import ImpactoAnual, ImpactoResumenItem, TasaPorVehiculo


# Limitaciones de nb06_resultados.json con redacción normalizada (mayúscula
# inicial, tildes, punto final), sin cambiar su contenido. La de "parque
# vehicular no medido" ya no es cierta: el proyecto cuenta con el parque real.
_LIMITACIONES_NORMALIZADAS = {
    "dataset individual sintetico; el analisis corre sobre agregados anuales oficiales": (
        "Dataset individual sintético; el análisis corre sobre agregados anuales oficiales."
    ),
    "sin contrafactual de mortalidad por desempeno insuficiente del modelo de severidad": (
        "Sin contrafactual de mortalidad por desempeño insuficiente del modelo de severidad."
    ),
    "2026 cubre 181 dias y excluye el periodo agostino": (
        "2026 cubre 181 días y excluye el período agostino."
    ),
    "sin proporcion de siniestros con alcohol involucrado": (
        "Sin proporción de siniestros con alcohol involucrado."
    ),
}
# Mes final de un período parcial, tal como lo declara `cobertura` en
# nb06_resultados.json ("parcial (ene-jun)"). El corte de parque del año
# parcial es el de fin de ESE mes, para que numerador (siniestros) y
# denominador (parque) cubran el mismo período.
_MESES_ABREVIADOS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}
_PATRON_COBERTURA_PARCIAL = re.compile(r"\(\s*[a-z]{3}\s*-\s*([a-z]{3})\s*\)", re.IGNORECASE)


def _fin_de_mes(anio: int, mes: int) -> date:
    siguiente = date(anio + (mes == 12), mes % 12 + 1, 1)
    return siguiente - timedelta(days=1)


def _corte_de_cobertura_parcial(anio: int, cobertura: str) -> date:
    coincidencia = _PATRON_COBERTURA_PARCIAL.search(cobertura)
    mes = _MESES_ABREVIADOS.get(coincidencia.group(1).lower()) if coincidencia else None
    if mes is None:
        raise ImpactoLeyRepositoryError(
            f"No se pudo leer el mes final de la cobertura parcial {cobertura!r} (año {anio})."
        )
    return _fin_de_mes(anio, mes)


# Desde este corte (ISO, comparable como texto) el distrito viene en cod_dis.
_PRIMER_CORTE_CON_COD_DIS = "2024-11-30"
_PREFIJO_PARQUE_NO_MEDIDO = "parque vehicular no medido"
_LIMITACION_PARQUE_REAL = (
    "El proyecto ya cuenta con el parque vehicular real (registro de ONASEVI); "
    "el análisis contrafactual no lo incorporó, por lo que su conclusión "
    "principal sigue siendo condicional al rango de crecimiento supuesto."
)


def _normalizar_limitacion(texto: str) -> str:
    if texto.startswith(_PREFIJO_PARQUE_NO_MEDIDO):
        return _LIMITACION_PARQUE_REAL
    if texto in _LIMITACIONES_NORMALIZADAS:
        return _LIMITACIONES_NORMALIZADAS[texto]
    return texto[:1].upper() + texto[1:]


class ImpactoLeyRepositoryError(RuntimeError):
    """`nb06_resultados.json` o el parque vehicular no existen o no tienen la forma esperada."""


class ImpactoLeyAdapter:
    def __init__(self, ruta_resultados: Path, ruta_parque: Path) -> None:
        self._ruta = ruta_resultados
        self._ruta_parque = ruta_parque
        self._tasas: list[TasaPorVehiculo] | None = None
        self._serie_anual: list[ImpactoAnual] | None = None
        self._fecha_aprobacion: date | None = None
        self._resumen: list[ImpactoResumenItem] | None = None
        self._limitaciones: list[str] | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta.exists():
            raise ImpactoLeyRepositoryError(f"No se encontró {self._ruta}")

        try:
            datos = json.loads(self._ruta.read_text(encoding="utf-8"))

            self._serie_anual = [
                ImpactoAnual(
                    anio=str(fila["anio"]),
                    siniestros=int(fila["siniestros"]),
                    fallecidos=int(fila["fallecidos"]),
                    letalidad=fila["fallecidos"] / fila["siniestros"] * 100,
                    cobertura=fila["cobertura"],
                )
                for fila in datos["fuente_oficial"]
            ]
            self._fecha_aprobacion = date.fromisoformat(datos["ley"]["aprobacion"])
            self._resumen = [
                ImpactoResumenItem(
                    metrica=fila["metrica"],
                    periodo=fila["periodo"],
                    variacion_interanual_pct=(
                        None
                        if _es_nan(fila["var_interanual_pct"])
                        else float(fila["var_interanual_pct"])
                    ),
                    lectura=fila["lectura"],
                )
                for fila in datos["resumen"]
            ]
            self._limitaciones = [_normalizar_limitacion(t) for t in datos["limitaciones"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ImpactoLeyRepositoryError(
                f"{self._ruta} no tiene la forma esperada: {exc}"
            ) from exc

        if not self._serie_anual:
            raise ImpactoLeyRepositoryError(f"{self._ruta}: fuente_oficial está vacía")

        self._tasas = self._calcular_tasas(datos["fuente_oficial"])

    def _calcular_tasas(self, fuente_oficial: list[dict]) -> list[TasaPorVehiculo]:
        """Siniestros por 1,000 vehículos y fallecidos por 10,000 vehículos,
        con el parque REAL (suma de todas las clases y distritos) al 31-dic
        de cada año; para el año parcial, el corte de fin del último mes
        cubierto (p. ej. «parcial (ene-jun)» -> 2026-06-30).

        Misma regla que src/config.load_parque_vehicular() (sin importarla):
        el distrito está en 'cod_mun' hasta el corte 2024-10-31 y en 'cod_dis'
        desde 2024-11-30, y se excluyen las filas sin código territorial
        (~0.1% del parque) para que el denominador coincida con el de los
        notebooks."""
        if not self._ruta_parque.exists():
            raise ImpactoLeyRepositoryError(f"No se encontró {self._ruta_parque}")
        try:
            parque = pd.read_csv(
                self._ruta_parque, usecols=["fecha_ref", "cod_mun", "cod_dis", "vehiculos"]
            )
        except ValueError as exc:
            raise ImpactoLeyRepositoryError(
                f"{self._ruta_parque.name} no tiene las columnas esperadas: {exc}"
            ) from exc
        nuevo = parque["fecha_ref"] >= _PRIMER_CORTE_CON_COD_DIS
        codigo = parque["cod_mun"].where(~nuevo, parque["cod_dis"])
        total_por_corte = parque[codigo.notna()].groupby("fecha_ref")["vehiculos"].sum()
        cortes = {date.fromisoformat(str(f)): int(v) for f, v in total_por_corte.items()}

        tasas: list[TasaPorVehiculo] = []
        for fila in fuente_oficial:
            anio = int(fila["anio"])
            parcial = fila["cobertura"] != "completo"
            if parcial:
                fin_periodo = _corte_de_cobertura_parcial(anio, fila["cobertura"])
                candidatos = [c for c in cortes if c == fin_periodo]
            else:
                candidatos = [c for c in cortes if c == date(anio, 12, 31)]
            if not candidatos:
                raise ImpactoLeyRepositoryError(
                    f"{self._ruta_parque.name}: no hay corte de parque para {anio}"
                )
            corte = max(candidatos)
            vehiculos = cortes[corte]
            tasas.append(
                TasaPorVehiculo(
                    anio=str(fila["anio"]),
                    parque_vehicular=vehiculos,
                    fecha_corte_parque=corte,
                    siniestros_por_1000_vehiculos=fila["siniestros"] / vehiculos * 1000,
                    fallecidos_por_10000_vehiculos=fila["fallecidos"] / vehiculos * 10000,
                    parcial=parcial,
                )
            )
        return tasas

    def _asegurar_cargado(self) -> None:
        if self._serie_anual is None:
            raise ImpactoLeyRepositoryError("El adaptador no fue validado (llamar cargar_y_validar).")

    def serie_anual(self) -> list[ImpactoAnual]:
        self._asegurar_cargado()
        assert self._serie_anual is not None
        return list(self._serie_anual)

    def fecha_aprobacion_ley(self) -> date:
        self._asegurar_cargado()
        assert self._fecha_aprobacion is not None
        return self._fecha_aprobacion

    def resumen(self) -> list[ImpactoResumenItem]:
        self._asegurar_cargado()
        assert self._resumen is not None
        return list(self._resumen)

    def tasas_por_vehiculo(self) -> list[TasaPorVehiculo]:
        self._asegurar_cargado()
        assert self._tasas is not None
        return list(self._tasas)

    def limitaciones(self) -> list[str]:
        self._asegurar_cargado()
        assert self._limitaciones is not None
        return list(self._limitaciones)


def _es_nan(valor: object) -> bool:
    return isinstance(valor, float) and valor != valor
