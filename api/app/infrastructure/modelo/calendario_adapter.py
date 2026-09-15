"""Adaptador de calendario: es_feriado (NB03 §5.2).

Nacional -> aplica a todos los distritos en esa fecha. Local (San Salvador)
-> aplica solo al distrito 'san salvador'. Ambas reglas se leen literalmente
de feriados_elsalvador_2022_2026.xlsx, sin agregar ni quitar fechas (NB03
"Ruta 1b").

**Fuera de 2022-2026** (aprobado explícitamente para este prototipo): se
recicla el patrón mes/día del último año presente en el archivo. Esto es
exacto para los feriados de fecha fija (Año Nuevo, 1 de mayo, Navidad, etc.)
y es una aproximación declarada para los móviles de Semana Santa (Jueves,
Viernes y Sábado Santo), que en la realidad no caen el mismo mes/día cada
año. Ver limitación documentada en api/README.md.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DISTRITO_LOCAL_SS = "san salvador"


class CalendarioRepositoryError(RuntimeError):
    """El archivo de feriados no existe o no tiene la forma esperada."""


class CalendarioAdapter:
    """Implementa CalendarioRepositoryPort."""

    def __init__(self, ruta_feriados: Path) -> None:
        self._ruta_feriados = ruta_feriados
        self._fechas_nacionales: set[date] | None = None
        self._fechas_local_ss: set[date] | None = None
        self._patron_nacional: set[tuple[int, int]] | None = None
        self._patron_local_ss: set[tuple[int, int]] | None = None
        self._anio_min: int | None = None
        self._anio_max: int | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta_feriados.exists():
            raise CalendarioRepositoryError(f"No se encontró el archivo de feriados en {self._ruta_feriados}")

        df = pd.read_excel(self._ruta_feriados)
        faltan = {"fecha", "tipo"} - set(df.columns)
        if faltan:
            raise CalendarioRepositoryError(
                f"Faltan columnas {sorted(faltan)} en {self._ruta_feriados.name}"
            )

        fechas = pd.to_datetime(df["fecha"]).dt.date
        tipo_norm = df["tipo"].astype(str).str.strip().str.lower()
        es_nacional = tipo_norm.str.startswith("nacional")
        es_local = tipo_norm.str.startswith("local")

        self._fechas_nacionales = set(fechas[es_nacional])
        self._fechas_local_ss = set(fechas[es_local])

        if not self._fechas_nacionales:
            raise CalendarioRepositoryError(
                f"No se encontraron feriados 'Nacional' en {self._ruta_feriados.name}"
            )

        todas = self._fechas_nacionales | self._fechas_local_ss
        self._anio_min = min(f.year for f in todas)
        self._anio_max = max(f.year for f in todas)

        self._patron_nacional = {(f.month, f.day) for f in self._fechas_nacionales if f.year == self._anio_max}
        self._patron_local_ss = {(f.month, f.day) for f in self._fechas_local_ss if f.year == self._anio_max}

        logger.info(
            "Calendario de feriados validado: %d nacionales, %d locales (SS), "
            "cobertura %d-%d (fuera de ese rango se recicla el patrón de %d)",
            len(self._fechas_nacionales),
            len(self._fechas_local_ss),
            self._anio_min,
            self._anio_max,
            self._anio_max,
        )

    def es_feriado(self, fecha: date, distrito_codigo: str) -> bool:
        if self._fechas_nacionales is None:
            raise CalendarioRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar) antes de usarse."
            )

        if self._anio_min <= fecha.year <= self._anio_max:
            nacional = fecha in self._fechas_nacionales
            local = fecha in self._fechas_local_ss and distrito_codigo == _DISTRITO_LOCAL_SS
        else:
            clave = (fecha.month, fecha.day)
            nacional = clave in self._patron_nacional
            local = clave in self._patron_local_ss and distrito_codigo == _DISTRITO_LOCAL_SS

        return bool(nacional or local)
