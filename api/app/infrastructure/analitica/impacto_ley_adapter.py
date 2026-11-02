"""Adaptador de impacto de la Ley Cero Tolerancia (NB06). Retrospectivo y
real: lee `nb06_resultados.json`, ya generado y verificado por NB06 — este
adaptador no recalcula nada, solo da forma al dato existente.

Se carga y valida una sola vez, en el lifespan (ver app/main.py), como el
resto de los adaptadores."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.domain.models import ImpactoAnual, ImpactoResumenItem


class ImpactoLeyRepositoryError(RuntimeError):
    """`nb06_resultados.json` no existe o no tiene la forma esperada."""


class ImpactoLeyAdapter:
    def __init__(self, ruta_resultados: Path) -> None:
        self._ruta = ruta_resultados
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
            self._limitaciones = list(datos["limitaciones"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ImpactoLeyRepositoryError(
                f"{self._ruta} no tiene la forma esperada: {exc}"
            ) from exc

        if not self._serie_anual:
            raise ImpactoLeyRepositoryError(f"{self._ruta}: fuente_oficial está vacía")

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

    def limitaciones(self) -> list[str]:
        self._asegurar_cargado()
        assert self._limitaciones is not None
        return list(self._limitaciones)


def _es_nan(valor: object) -> bool:
    return isinstance(valor, float) and valor != valor
