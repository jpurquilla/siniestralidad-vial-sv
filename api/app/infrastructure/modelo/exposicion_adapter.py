"""Adaptador de exposición: log_exposicion_v2 = log_exposicion (NB03, base
por distrito) + ln(factor_crec(fecha)) (NB05, tendencia CAGR).

log_exposicion_v2 es la pieza que junto con log_tasa_base (ver
xgboost_adapter.py) compone el base_margin del modelo de frecuencia. La base
por distrito se lee de matriz_frecuencia_v2.csv (es constante por distrito
en todo el histórico); g_mensual y el origen temporal de la tendencia se leen
de nb05_especificacion_v2.json, nunca se hardcodean.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ExposicionRepositoryError(RuntimeError):
    """La especificación de tendencia (NB05) o la matriz de exposición base
    (NB03) no pasaron las validaciones de arranque."""


class ExposicionAdapter:
    """Implementa ExposicionRepositoryPort."""

    def __init__(self, ruta_matriz: Path, ruta_especificacion_nb05: Path) -> None:
        self._ruta_matriz = ruta_matriz
        self._ruta_especificacion_nb05 = ruta_especificacion_nb05
        self._log_exposicion_base: dict[str, float] | None = None
        self.g_mensual: float | None = None
        self.origen_anio: int | None = None
        self.origen_mes: int | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta_matriz.exists():
            raise ExposicionRepositoryError(f"No se encontró la matriz en {self._ruta_matriz}")
        if not self._ruta_especificacion_nb05.exists():
            raise ExposicionRepositoryError(
                f"No se encontró la especificación en {self._ruta_especificacion_nb05}"
            )

        try:
            especificacion = json.loads(self._ruta_especificacion_nb05.read_text(encoding="utf-8"))
            tendencia = especificacion["correccion_tendencia"]
            g_mensual = float(tendencia["g_mensual"])
            origen_temporal = tendencia["origen_temporal"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ExposicionRepositoryError(
                f"{self._ruta_especificacion_nb05.name} no tiene la forma esperada "
                f"en 'correccion_tendencia': {exc}"
            ) from exc

        if not math.isfinite(g_mensual) or not (-1 < g_mensual < 1):
            raise ExposicionRepositoryError(
                f"g_mensual leído ({g_mensual}) no es una tasa de crecimiento mensual plausible."
            )

        try:
            origen = date.fromisoformat(str(origen_temporal)[:10])
        except ValueError as exc:
            raise ExposicionRepositoryError(
                f"origen_temporal ({origen_temporal!r}) no es una fecha ISO válida."
            ) from exc

        df = pd.read_csv(
            self._ruta_matriz,
            usecols=["distrito_key", "log_exposicion"],
            dtype={"distrito_key": str, "log_exposicion": float},
        )
        n_valores_por_distrito = df.groupby("distrito_key")["log_exposicion"].nunique()
        inconsistentes = n_valores_por_distrito[n_valores_por_distrito > 1]
        if not inconsistentes.empty:
            raise ExposicionRepositoryError(
                "log_exposicion no es constante por distrito en "
                f"{self._ruta_matriz.name} para: {list(inconsistentes.index)}"
            )

        self._log_exposicion_base = (
            df.drop_duplicates("distrito_key").set_index("distrito_key")["log_exposicion"].to_dict()
        )
        self.g_mensual = g_mensual
        self.origen_anio = origen.year
        self.origen_mes = origen.month

        logger.info(
            "Exposición validada: %d distritos, g_mensual=%.8f, origen=%04d-%02d",
            len(self._log_exposicion_base),
            g_mensual,
            origen.year,
            origen.month,
        )

    def _meses_desde_origen(self, fecha: date) -> int:
        return (fecha.year - self.origen_anio) * 12 + (fecha.month - self.origen_mes)

    def log_exposicion_v2(self, distrito_codigo: str, fecha: date) -> float:
        if self._log_exposicion_base is None:
            raise ExposicionRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar) antes de usarse."
            )
        try:
            base = self._log_exposicion_base[distrito_codigo]
        except KeyError as exc:
            raise ExposicionRepositoryError(
                f"Distrito desconocido en la matriz de exposición: {distrito_codigo!r}"
            ) from exc

        meses = self._meses_desde_origen(fecha)
        ln_factor_crec = meses * math.log1p(self.g_mensual)
        return base + ln_factor_crec
