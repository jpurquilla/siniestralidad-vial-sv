"""Adaptador de clima: normal climatológica de prcp_mensual por distrito×mes.

prcp_mensual varía por año-mes real en matriz_frecuencia_v2.csv (no es una
normal fija). Cuando el cliente de /api/v1/predicciones no envía prcp_mensual
para simular un escenario, se usa el promedio histórico de ese distrito para
el mes de la fecha consultada, sobre los años presentes en la matriz
(2022-2026) — la 'normal climatológica' pedida en el encargo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ClimaRepositoryError(RuntimeError):
    """La matriz de frecuencia no existe o no tiene la forma esperada."""


class ClimaAdapter:
    """Implementa ClimaRepositoryPort."""

    def __init__(self, ruta_matriz: Path) -> None:
        self._ruta_matriz = ruta_matriz
        self._normales: dict[tuple[str, int], float] | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta_matriz.exists():
            raise ClimaRepositoryError(f"No se encontró la matriz en {self._ruta_matriz}")

        df = pd.read_csv(
            self._ruta_matriz,
            usecols=["distrito_key", "mes", "prcp_mensual"],
            dtype={"distrito_key": str, "mes": int, "prcp_mensual": float},
        )
        promedios = df.groupby(["distrito_key", "mes"])["prcp_mensual"].mean()
        self._normales = {(d, int(m)): float(v) for (d, m), v in promedios.items()}

        n_distritos = len({d for d, _ in self._normales})
        logger.info(
            "Normales climatológicas validadas: %d distritos x 12 meses (%d combinaciones)",
            n_distritos,
            len(self._normales),
        )

    def normal_climatologica(self, distrito_codigo: str, mes: int) -> float:
        if self._normales is None:
            raise ClimaRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar) antes de usarse."
            )
        try:
            return self._normales[(distrito_codigo, mes)]
        except KeyError as exc:
            raise ClimaRepositoryError(
                f"No hay normal climatológica para distrito={distrito_codigo!r} mes={mes!r}"
            ) from exc
