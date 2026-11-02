"""Adaptador del crecimiento del parque vehicular a 12 meses (crec_parque_12m).

La variable vive en data/processed/matriz_frecuencia_v3.csv.gz y es única por
(distrito_key, anio_mes). Cubre 2022-01..2026-06. Para fechas fuera de ese
rango se usa el mes disponible más cercano del distrito (el último, para
fechas futuras; el primero, para anteriores a 2022-01) y la respuesta lo
declara con FuenteParque.ULTIMO_DISPONIBLE, para que ninguna predicción
extrapolada pase por observada.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from app.domain.models import FuenteParque

logger = logging.getLogger(__name__)


class ParqueRepositoryError(RuntimeError):
    """La matriz no existe, no tiene la forma esperada o crec_parque_12m no
    es único por (distrito, mes)."""


class ParqueAdapter:
    """Implementa ParqueRepositoryPort."""

    def __init__(self, ruta_matriz: Path) -> None:
        self._ruta_matriz = ruta_matriz
        self._por_distrito: dict[str, dict[str, float]] | None = None
        self.mes_minimo: str | None = None  # "YYYY-MM"
        self.mes_maximo: str | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta_matriz.exists():
            raise ParqueRepositoryError(f"No se encontró la matriz en {self._ruta_matriz}")

        try:
            df = pd.read_csv(
                self._ruta_matriz,
                usecols=["distrito_key", "anio_mes", "crec_parque_12m"],
                dtype={"distrito_key": str, "anio_mes": str, "crec_parque_12m": float},
            )
        except ValueError as exc:
            raise ParqueRepositoryError(
                f"{self._ruta_matriz.name} no tiene las columnas esperadas: {exc}"
            ) from exc

        if df["crec_parque_12m"].isna().any():
            raise ParqueRepositoryError("crec_parque_12m tiene valores nulos en la matriz.")
        n_valores = df.groupby(["distrito_key", "anio_mes"])["crec_parque_12m"].nunique()
        no_unicos = n_valores[n_valores > 1]
        if not no_unicos.empty:
            raise ParqueRepositoryError(
                "crec_parque_12m no es único por (distrito, mes) en "
                f"{self._ruta_matriz.name}: {len(no_unicos)} combinaciones, "
                f"p. ej. {no_unicos.index[0]}"
            )

        unico = df.drop_duplicates(["distrito_key", "anio_mes"])
        por_distrito: dict[str, dict[str, float]] = {}
        for fila in unico.itertuples(index=False):
            por_distrito.setdefault(fila.distrito_key, {})[fila.anio_mes] = float(fila.crec_parque_12m)

        meses_por_distrito = {frozenset(m) for m in por_distrito.values()}
        if len(meses_por_distrito) != 1:
            raise ParqueRepositoryError("Los distritos no comparten el mismo conjunto de meses.")

        meses = sorted(next(iter(meses_por_distrito)))
        self._por_distrito = por_distrito
        self.mes_minimo, self.mes_maximo = meses[0], meses[-1]
        logger.info(
            "Parque vehicular validado: %d distritos, meses %s..%s",
            len(por_distrito),
            self.mes_minimo,
            self.mes_maximo,
        )

    def crec_parque_12m(self, distrito_codigo: str, fecha: date) -> tuple[float, FuenteParque]:
        if self._por_distrito is None or self.mes_minimo is None or self.mes_maximo is None:
            raise ParqueRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar) antes de usarse."
            )
        try:
            serie = self._por_distrito[distrito_codigo]
        except KeyError as exc:
            raise ParqueRepositoryError(
                f"Distrito desconocido en la serie de parque: {distrito_codigo!r}"
            ) from exc

        mes = f"{fecha.year:04d}-{fecha.month:02d}"
        if mes in serie:
            return serie[mes], FuenteParque.OBSERVADO
        # Fuera de rango: el mes disponible más cercano (los meses son strings
        # ISO, así que el orden lexicográfico es el cronológico).
        mes_cercano = self.mes_minimo if mes < self.mes_minimo else self.mes_maximo
        return serie[mes_cercano], FuenteParque.ULTIMO_DISPONIBLE
