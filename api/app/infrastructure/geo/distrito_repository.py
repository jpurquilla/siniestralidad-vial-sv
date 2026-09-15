"""Catálogo de los 103 distritos: código, nombre, departamento, pct_urbano,
población. Se construye una sola vez, al arrancar, cruzando dos fuentes:

- matriz_frecuencia_v2.csv: distrito_key (código normalizado, sin tildes),
  departamento_key, pct_urbano, poblacion, poblacion_baja — un valor
  constante por distrito (NB03).
- data/raw/siniestros.csv: nombre y departamento tal como se escriben en la
  fuente original (con tildes), para mostrarlos legibles en la API.

La normalización (minúsculas, sin tildes) replica src/config.norm_distrito()
del proyecto raíz — se reimplementa aquí, sin importar src/, para que api/
sea un servicio autocontenido con su propio requirements.txt.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from app.domain.models import Distrito

# Corrección de nomenclatura fuente->censo (idéntica a
# src/config.MAPEO_DISTRITOS): "Dolores" en siniestros.csv es
# "Villa Dolores" en el resto de fuentes (Cabañas, reforma territorial 2023).
_MAPEO_DISTRITOS = {"dolores": "villa dolores"}


def _normalizar(s: str) -> str:
    s = str(s).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _reconciliar(s: str) -> str:
    n = _normalizar(s)
    return _MAPEO_DISTRITOS.get(n, n)


class DistritoRepositoryError(RuntimeError):
    """Las fuentes del catálogo de distritos no existen o no tienen la forma esperada."""


class DistritoRepository:
    """Implementa DistritoRepositoryPort."""

    def __init__(self, ruta_matriz: Path, ruta_siniestros: Path) -> None:
        self._ruta_matriz = ruta_matriz
        self._ruta_siniestros = ruta_siniestros
        self._distritos: dict[str, Distrito] | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta_matriz.exists():
            raise DistritoRepositoryError(f"No se encontró la matriz en {self._ruta_matriz}")
        if not self._ruta_siniestros.exists():
            raise DistritoRepositoryError(f"No se encontró {self._ruta_siniestros}")

        base = pd.read_csv(
            self._ruta_matriz,
            usecols=["distrito_key", "departamento_key", "pct_urbano", "poblacion", "poblacion_baja"],
        ).drop_duplicates("distrito_key")

        crudos = pd.read_csv(
            self._ruta_siniestros, sep=";", usecols=["distrito", "departamento"]
        ).drop_duplicates()
        crudos["distrito_key"] = crudos["distrito"].map(_reconciliar)
        crudos["departamento_key"] = crudos["departamento"].map(_normalizar)
        nombres = crudos.drop_duplicates("distrito_key").set_index("distrito_key")

        faltan = set(base["distrito_key"]) - set(nombres.index)
        if faltan:
            raise DistritoRepositoryError(
                f"{len(faltan)} distritos de la matriz no tienen nombre en "
                f"{self._ruta_siniestros.name}: {sorted(faltan)}"
            )

        distritos: dict[str, Distrito] = {}
        for fila in base.itertuples(index=False):
            nombre_fila = nombres.loc[fila.distrito_key]
            distritos[fila.distrito_key] = Distrito(
                codigo=fila.distrito_key,
                nombre=nombre_fila["distrito"],
                departamento=nombre_fila["departamento"],
                pct_urbano=float(fila.pct_urbano),
                poblacion=int(fila.poblacion),
                poblacion_baja=bool(fila.poblacion_baja),
            )

        if len(distritos) != 103:
            raise DistritoRepositoryError(
                f"Se esperaban 103 distritos y se construyeron {len(distritos)}."
            )

        self._distritos = distritos

    def listar(self) -> list[Distrito]:
        self._asegurar_cargado()
        return list(self._distritos.values())

    def obtener(self, codigo: str) -> Distrito | None:
        self._asegurar_cargado()
        return self._distritos.get(codigo)

    def existe(self, codigo: str) -> bool:
        self._asegurar_cargado()
        return codigo in self._distritos

    def _asegurar_cargado(self) -> None:
        if self._distritos is None:
            raise DistritoRepositoryError(
                "El repositorio no fue validado (llamar cargar_y_validar) antes de usarse."
            )
