"""Catálogo de los 103 distritos: código, nombre, departamento, pct_urbano,
población y las dos variables censales del modelo. Se construye una sola
vez, al arrancar, cruzando dos fuentes:

- matriz_frecuencia_v3.csv.gz: distrito_key (código normalizado, sin tildes),
  departamento_key, pct_urbano, poblacion, poblacion_baja, pct_hogares_auto y
  pct_trabaja_otro_lugar — un valor constante por distrito (se valida).
- data/raw/TAB_POB_AREA_1.xlsx (Censo 2024): nombre y departamento con
  tildes, para mostrarlos legibles. Se lee con la misma lógica que
  src/config.load_censo() (sin importar src/) y se cruza por
  (distrito_key, departamento_key): hay nombres repetidos entre departamentos.

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
# El censo ya usa "Villa Dolores"; se conserva la reconciliación por si la
# fuente usara el nombre antiguo.
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

    def __init__(self, ruta_matriz: Path, ruta_censo: Path) -> None:
        self._ruta_matriz = ruta_matriz
        self._ruta_censo = ruta_censo
        self._distritos: dict[str, Distrito] | None = None
        self._nombre_por_clave: dict[tuple[str, str], str] = {}

    def _leer_nombres_censo(self) -> pd.DataFrame:
        """Nombres de los 262 distritos del censo (réplica de
        src/config.load_censo(): saltar 6 filas de encabezado, descartar
        subtotales 'TOTAL', 'No Especificado' y la nota al pie, y quitar el
        prefijo 'NN - ')."""
        df = pd.read_excel(self._ruta_censo, header=None, skiprows=6)
        df.columns = ["departamento", "municipio", "distrito", "total", "urb_h", "urb_m", "rur_h", "rur_m"]
        mascara = (
            df["distrito"].notna()
            & (~df["distrito"].astype(str).str.contains("TOTAL", na=False))
            & (df["departamento"].astype(str).str.match(r"^\d"))
        )
        d = df[mascara].copy()
        for columna in ["departamento", "distrito"]:
            d[columna] = d[columna].astype(str).str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()
        d["distrito_key"] = d["distrito"].map(_reconciliar)
        d["departamento_key"] = d["departamento"].map(_normalizar)
        return d[["distrito_key", "departamento_key", "distrito", "departamento"]]

    def cargar_y_validar(self) -> None:
        if not self._ruta_matriz.exists():
            raise DistritoRepositoryError(f"No se encontró la matriz en {self._ruta_matriz}")
        if not self._ruta_censo.exists():
            raise DistritoRepositoryError(f"No se encontró {self._ruta_censo}")

        columnas = [
            "distrito_key",
            "departamento_key",
            "pct_urbano",
            "poblacion",
            "poblacion_baja",
            "pct_hogares_auto",
            "pct_trabaja_otro_lugar",
        ]
        crudo = pd.read_csv(self._ruta_matriz, usecols=columnas)
        valores_por_distrito = crudo.groupby("distrito_key")[columnas[1:]].nunique()
        no_constantes = valores_por_distrito[(valores_por_distrito > 1).any(axis=1)]
        if not no_constantes.empty:
            raise DistritoRepositoryError(
                f"Variables de distrito no constantes en {self._ruta_matriz.name} "
                f"para: {sorted(no_constantes.index)}"
            )
        base = crudo.drop_duplicates("distrito_key")

        nombres = self._leer_nombres_censo().drop_duplicates(["distrito_key", "departamento_key"])
        self._nombre_por_clave = {
            (f.distrito_key, f.departamento_key): f.distrito for f in nombres.itertuples(index=False)
        }
        cruce = base.merge(nombres, on=["distrito_key", "departamento_key"], how="left")
        faltan = cruce[cruce["distrito"].isna()]["distrito_key"]
        if len(faltan):
            raise DistritoRepositoryError(
                f"{len(faltan)} distritos de la matriz no tienen nombre en "
                f"{self._ruta_censo.name}: {sorted(faltan)}"
            )

        distritos: dict[str, Distrito] = {}
        for fila in cruce.itertuples(index=False):
            distritos[fila.distrito_key] = Distrito(
                codigo=fila.distrito_key,
                nombre=fila.distrito,
                departamento=fila.departamento,
                pct_urbano=float(fila.pct_urbano),
                poblacion=int(fila.poblacion),
                poblacion_baja=bool(fila.poblacion_baja),
                pct_hogares_auto=float(fila.pct_hogares_auto),
                pct_trabaja_otro_lugar=float(fila.pct_trabaja_otro_lugar),
            )

        if len(distritos) != 103:
            raise DistritoRepositoryError(
                f"Se esperaban 103 distritos y se construyeron {len(distritos)}."
            )

        self._distritos = distritos

    def nombre_para_mostrar(self, distrito_crudo: str, departamento_crudo: str) -> str:
        """Nombre con tildes del censo para un distrito escrito como en
        `siniestros.csv` (sin tildes). Si no hay coincidencia devuelve el
        texto original, nunca inventa un nombre."""
        self._asegurar_cargado()
        clave = (_reconciliar(distrito_crudo), _normalizar(departamento_crudo))
        return self._nombre_por_clave.get(clave, distrito_crudo)

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
