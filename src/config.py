"""
config.py — Configuración central del proyecto siniestralidad-vial-sv.

Punto único de verdad para:
  - Semilla global de reproducibilidad (SEED = 42)
  - Rutas del proyecto (raw, processed, models, reports, docs)
  - Cargadores de datos (load_siniestros, load_victimas, load_censo, ...)
  - Puente carpeta-de-clima -> estación del catálogo (coordenadas GPS)

Regla del proyecto: todos los notebooks importan este módulo
    from src import config as cfg
y NO redefinen rutas ni semilla.

Alcance del módulo: SOLO carga y normalización de formato (plomería).
Las decisiones analíticas (descartar estaciones, elegir variables, limpieza
por calidad de datos) viven en los notebooks (NB01/NB02/NB03), no aquí.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 2. Rutas del proyecto (relativas a este archivo -> portables)
# ---------------------------------------------------------------------------
# src/config.py  ->  la raíz del proyecto es el padre de src/
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # se versiona en este proyecto
PROCESSED_DIR = DATA_DIR / "processed"  # se genera en NB03 (se versiona)
CLIMA_DIR = RAW_DIR / "clima"  # estaciones Meteostat horarias

MODELS_DIR = ROOT_DIR / "models"  # .pkl (se versionan: entregables)
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"  # métricas .json
DOCS_DIR = ROOT_DIR / "docs"

# Salidas: se crean si no existen. raw/ NO se crea (debe llenarse a mano).
for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Archivos de datos (materia prima Etapa 1)
# ---------------------------------------------------------------------------
SINIESTROS_FILE = RAW_DIR / "siniestros.csv"
VICTIMAS_FILE = RAW_DIR / "victimas.csv"
FERIADOS_FILE = RAW_DIR / "feriados_elsalvador_2022_2026.xlsx"

# Censo: archivo OFICIAL CRUDO del VII Censo de Población 2024 (BCR/ONEC),
# tabla TAB_POB_AREA. Se limpia en load_censo() (ver más abajo).
CENSO_FILE = RAW_DIR / "TAB_POB_AREA_1.xlsx"

# Catálogo de estaciones meteorológicas (id WMO + nombre + coordenadas GPS).
# Columnas: id, nombre, pais, region, latitud, longitud, elevacion, timezone.
# Necesario en NB03 para el join "estación más cercana" a cada distrito.
ESTACIONES_FILE = RAW_DIR / "Estaciones SLV.xlsx"

# Estaciones de clima: clave lógica -> subcarpeta dentro de clima/.
# El cargador NO asume nombre de archivo: lee todos los .csv de la carpeta.
# Dos formatos internos conviven (se unifican en _leer_estacion):
#   - IL, SA, SO   -> archivo único, esquema con 'date'+'datetime'
#   - PAZ, SM, SS  -> un CSV por año, esquema con 'year/month/day/hour'
# NOTA: la evaluación de calidad (p. ej. completitud de prcp en SS) y la
# decisión de qué estación usar se hacen en el EDA (NB02), no aquí.
ESTACIONES_CLIMA = {
    "IL": "IL",  # 78663 San Salvador / Ilopango (AMSS)
    "SA": "SA",  # 78655 Santa Ana / El Palmar
    "SO": "SO",  # 78650 Acajutla (región SO)
    "PAZ": "PAZ",  # 78666 Aeropuerto Intl. / Comalapa (región PZ)
    "SM": "SM",  # 78670 San Miguel / El Papalón
    "SS": "SS",  # 78662 San Salvador (centro)
    # "UN": "UN",  # 78672 La Unión — carpeta vacía, sin datos usables
}

# Puente carpeta-de-clima -> id WMO en el catálogo (Estaciones SLV).
# Permite recuperar las coordenadas GPS de cada estación desde load_estaciones().
CLIMA_A_ESTACION = {
    "IL": 78663,  # San Salvador / Ilopango (AMSS)
    "SA": 78655,  # Santa Ana / El Palmar
    "SO": 78650,  # Acajutla (región SO)
    "PAZ": 78666,  # Aeropuerto Intl. / Comalapa (región PZ)
    "SM": 78670,  # San Miguel / El Papalón
    "SS": 78662,  # San Salvador (centro)
    # "UN": 78672, # La Unión — catálogo tiene coords pero carpeta de clima vacía
}


# ---------------------------------------------------------------------------
# 4. Cargadores de datos
# ---------------------------------------------------------------------------
def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Lee un CSV con mensaje de error claro si falta.
    Separador por defecto ';' (los CSV se generaron con config regional ES);
    se puede sobreescribir pasando sep= en kwargs.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path.name} en {path.parent}.")
    kwargs.setdefault("sep", ";")
    return pd.read_csv(path, **kwargs)


def load_siniestros() -> pd.DataFrame:
    """
    89.946 accidentes (CSV separado por ';').

    'fecha' se parsea como datetime y se agrega 'fecha_hora' (fecha + hora)
    como timestamp completo para análisis temporal.
    """
    df = _read_csv(SINIESTROS_FILE, parse_dates=["fecha"])
    if "hora" in df.columns:
        df["fecha_hora"] = pd.to_datetime(
            df["fecha"].dt.strftime("%Y-%m-%d") + " " + df["hora"].astype(str),
            errors="coerce",
        )
    return df


def load_victimas() -> pd.DataFrame:
    """60.402 víctimas, enlazadas a siniestros por id_siniestro (CSV con ';')."""
    return _read_csv(VICTIMAS_FILE, parse_dates=["fecha"])


def load_censo() -> pd.DataFrame:
    """
    VII Censo de Población 2024 (BCR/ONEC), tabla TAB_POB_AREA.

    Devuelve el nivel DISTRITO ya estructurado: 262 distritos con población
    total, urbana, rural y % urbano. Descarta los subtotales 'TOTAL', la fila
    'No Especificado' y la nota de fuente al pie del archivo.

    Esto es normalización de formato del archivo oficial (multi-encabezado,
    prefijos numéricos), no análisis. La coincidencia de nombres de distrito
    con siniestros.csv se valida en NB03 (clave del join per cápita).
    """
    if not CENSO_FILE.exists():
        raise FileNotFoundError(f"No se encontró {CENSO_FILE.name} en {RAW_DIR}.")

    df = pd.read_excel(CENSO_FILE, header=None, skiprows=6)
    df.columns = [
        "departamento",
        "municipio",
        "distrito",
        "total",
        "urb_h",
        "urb_m",
        "rur_h",
        "rur_m",
    ]

    # Filtrar a nivel distrito real:
    #   - distrito no nulo y distinto de 'TOTAL' (descarta subtotales)
    #   - departamento empieza con dígito (descarta 'No Especificado' y footnote)
    mask = (
        df["distrito"].notna()
        & (~df["distrito"].astype(str).str.contains("TOTAL", na=False))
        & (df["departamento"].astype(str).str.match(r"^\d"))
    )
    d = df[mask].copy()

    # Numéricos: NaN en columnas urbano/rural = 0 (distritos 100% rurales, etc.)
    for c in ["total", "urb_h", "urb_m", "rur_h", "rur_m"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    d["poblacion"] = d["total"].astype(int)
    d["urbano"] = (d["urb_h"] + d["urb_m"]).astype(int)
    d["rural"] = (d["rur_h"] + d["rur_m"]).astype(int)
    d["pct_urbano"] = (d["urbano"] / d["poblacion"]).round(4)

    # Quitar prefijo numérico "NN - " de los nombres
    for c in ["departamento", "municipio", "distrito"]:
        d[c] = d[c].astype(str).str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()

    cols = [
        "departamento",
        "municipio",
        "distrito",
        "poblacion",
        "urbano",
        "rural",
        "pct_urbano",
    ]
    return d[cols].reset_index(drop=True)


def load_estaciones() -> pd.DataFrame:
    """
    Catálogo de estaciones meteorológicas: id WMO, nombre y coordenadas GPS.
    Columnas: id, nombre, pais, region, latitud, longitud, elevacion, timezone.
    """
    if not ESTACIONES_FILE.exists():
        raise FileNotFoundError(f"No se encontró {ESTACIONES_FILE.name} en {RAW_DIR}.")
    return pd.read_excel(ESTACIONES_FILE)


def estaciones_con_datos() -> pd.DataFrame:
    """
    Estaciones que tienen carpeta de clima con archivos, cruzadas con el
    catálogo para traer sus coordenadas. Base para el join 'estación más
    cercana' de NB03.

    Devuelve: clima_key, id, nombre, latitud, longitud.
    NO aplica criterios de calidad (p. ej. excluir SS por baja completitud);
    esa decisión se toma explícitamente en el EDA (NB02).
    """
    cat = load_estaciones().set_index("id")
    filas = []
    for folder, wmo_id in CLIMA_A_ESTACION.items():
        carpeta = CLIMA_DIR / ESTACIONES_CLIMA.get(folder, folder)
        if _clima_files(carpeta) and wmo_id in cat.index:
            r = cat.loc[wmo_id]
            filas.append(
                {
                    "clima_key": folder,
                    "id": wmo_id,
                    "nombre": r["nombre"],
                    "latitud": float(r["latitud"]),
                    "longitud": float(r["longitud"]),
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 3.5 Normalización de claves geográficas (nombres de distrito)
# ---------------------------------------------------------------------------
# El nombre de distrito NO coincide literalmente entre fuentes:
#   - siniestros.csv escribe los nombres SIN tildes ("Ahuachapan", "Colon")
#   - el censo (TAB_POB_AREA) los conserva CON tildes ("Ahuachapán", "Colón")
# Un join literal perdería ~22 distritos. norm_distrito() lleva ambos lados a
# una forma canónica (minúsculas, sin tildes, sin espacios extremos).
#
# Además, un distrito difiere por nomenclatura (no solo por tildes): la fuente
# de siniestros abrevia "Villa Dolores" (Cabañas Este) como "Dolores". Los
# casos de este tipo se corrigen con MAPEO_DISTRITOS, aplicado DESPUÉS de
# normalizar. reconciliar_distrito() combina ambos pasos en uno.
#
# Esto es normalización de formato (plomería), reutilizable por NB03 y NB04.
# La CLAVE de join y su validación de cobertura se deciden en el notebook.

# Correcciones de nomenclatura fuente->censo, en forma YA NORMALIZADA.
# Clave = nombre normalizado tal como aparece en siniestros.
# Valor = nombre normalizado tal como aparece en el censo.
MAPEO_DISTRITOS = {
    "dolores": "villa dolores",  # Cabañas, Cabañas Este (reforma territorial 2023)
}


def norm_distrito(s: str) -> str:
    """
    Normaliza un nombre de distrito a forma canónica para el join:
    minúsculas, sin tildes/diacríticos, sin espacios extremos.
    Devuelve el valor tal cual si es NaN/None (no fuerza a str).
    """
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    return s


def reconciliar_distrito(s: str) -> str:
    """
    Normaliza (norm_distrito) y luego aplica MAPEO_DISTRITOS para corregir
    diferencias de nomenclatura entre fuentes. Es la función a usar como
    clave de join distrito<->censo/clima en NB03.
    """
    n = norm_distrito(s)
    if pd.isna(n):
        return n
    return MAPEO_DISTRITOS.get(n, n)


def load_feriados() -> pd.DataFrame:
    """Calendario de feriados de El Salvador 2022–2026."""
    if not FERIADOS_FILE.exists():
        raise FileNotFoundError(f"Falta {FERIADOS_FILE.name} en {RAW_DIR}.")
    return pd.read_excel(FERIADOS_FILE)


def _clima_files(carpeta: Path) -> list[Path]:
    """Todos los .csv de una estación, ordenados (año asc si aplica)."""
    return sorted(carpeta.glob("*.csv"))


# Columnas crudas de tiempo que se consolidan en 'time' y luego se descartan.
# (esto es unificación de formato, no selección analítica de variables)
_COLS_TIEMPO_CRUDAS = ["year", "month", "day", "hour", "date", "datetime"]


def _leer_estacion(carpeta: Path) -> pd.DataFrame:
    """
    Lee una estación y unifica SOLO el formato de tiempo. Plomería, no análisis.

    Concatena todos los .csv de la carpeta, construye una columna 'time'
    (datetime) a partir del esquema que traiga el archivo, y descarta las
    columnas crudas de tiempo (ya representadas por 'time'). TODAS las demás
    columnas de datos se conservan tal cual: la decisión de qué variables usar
    o descartar se toma en los notebooks.

    Esquemas de tiempo soportados:
      - 'year/month/day/hour'  (PAZ, SM, SS)
      - 'datetime'             (IL, SA, SO)
      - 'date' + 'hour'        (respaldo)
    """
    files = _clima_files(carpeta)
    if not files:
        raise FileNotFoundError(f"No hay .csv en {carpeta}.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # --- Construir 'time' según el formato disponible ---
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif {"date", "hour"}.issubset(df.columns):
        df["time"] = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(
            df["hour"], unit="h"
        )
    else:
        raise ValueError(
            f"No pude construir 'time' en {carpeta.name}. "
            f"Columnas: {df.columns.tolist()}"
        )

    # Descartar columnas crudas de tiempo (ya consolidadas en 'time')
    df = df.drop(columns=[c for c in _COLS_TIEMPO_CRUDAS if c in df.columns])

    # 'time' al frente, resto de columnas intactas
    otras = [c for c in df.columns if c != "time"]
    df = df[["time"] + otras].sort_values("time").reset_index(drop=True)
    return df


def load_clima(estacion: str) -> pd.DataFrame:
    """Clima horario de una estación (clave en ESTACIONES_CLIMA)."""
    if estacion not in ESTACIONES_CLIMA:
        raise KeyError(
            f"Estación '{estacion}' no registrada. Disponibles: {list(ESTACIONES_CLIMA)}"
        )
    return _leer_estacion(CLIMA_DIR / ESTACIONES_CLIMA[estacion])


def load_clima_all() -> pd.DataFrame:
    """
    Todas las estaciones concatenadas, con columna 'estacion'.

    Como los dos formatos tienen columnas de datos distintas (p. ej. IL/SA/SO
    traen dwpt/snow/tsun; PAZ/SM/SS traen cldc y *_source), el concat resultante
    tendrá NaN donde una columna no existe en cierto formato. Eso es esperado y
    honesto; el EDA (NB02) decide qué variables conservar.
    """
    frames = []
    for clave, sub in ESTACIONES_CLIMA.items():
        carpeta = CLIMA_DIR / sub
        if _clima_files(carpeta):
            df = _leer_estacion(carpeta)
            df["estacion"] = clave
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No hay archivos de clima en {CLIMA_DIR}.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Helpers de salida (figuras y métricas) — usados por NB02 y NB04
# ---------------------------------------------------------------------------
def save_fig(fig, nombre: str, dpi: int = 150) -> Path:
    """Guarda una figura en reports/figures/ y devuelve la ruta."""
    path = FIGURES_DIR / (nombre if nombre.endswith(".png") else f"{nombre}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_metrics(metrics: dict, nombre: str) -> Path:
    """Guarda métricas como .json en reports/results/."""
    path = RESULTS_DIR / (nombre if nombre.endswith(".json") else f"{nombre}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def load_metrics(nombre: str) -> dict:
    """Lee métricas .json de reports/results/."""
    path = RESULTS_DIR / (nombre if nombre.endswith(".json") else f"{nombre}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 6. Chequeo rápido:  python -m src.config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"ROOT_DIR = {ROOT_DIR}")
    print(f"SEED     = {SEED}\n")

    base = {
        "siniestros.csv": SINIESTROS_FILE,
        "victimas.csv": VICTIMAS_FILE,
        "TAB_POB_AREA_1.xlsx (censo)": CENSO_FILE,
        "Estaciones SLV.xlsx": ESTACIONES_FILE,
        "feriados_*.xlsx": FERIADOS_FILE,
    }
    for nombre, path in base.items():
        print(f"[{'OK ' if path.exists() else 'FALTA'}] {nombre}")

    print()
    for clave, sub in ESTACIONES_CLIMA.items():
        files = _clima_files(CLIMA_DIR / sub)
        estado = f"OK ({len(files)} arch.)" if files else "FALTA"
        print(f"[{estado}] clima/{sub}")
