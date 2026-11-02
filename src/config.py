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
GEO_DIR = DATA_DIR / "geo"  # geometrías (GeoJSON)
# Copias crudas de las fuentes web que no ofrecen archivo descargable
# (respuestas del Power BI de ONASEVI, páginas consultadas para el calendario de fiestas).
# Las generan los scripts de src/extraccion/; los notebooks no las leen.
EXTRACCION_DIR = DATA_DIR / "fuente_original" / "extraccion"

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

# --- Fuentes externas de la etapa final -------------------------------------
# Censo 2024 (BCR/ONEC), tabulados OFICIALES CRUDOS, mismo origen que CENSO_FILE:
#   TAB_HOG_4: hogares según disponibilidad de automóvil, bicicleta y motocicleta.
#   TAB_CE_5 : población ocupada según lugar de trabajo (pregunta P22).
CENSO_BIENES_FILE = RAW_DIR / "TAB_HOG_4.xlsx"
CENSO_TRABAJO_FILE = RAW_DIR / "TAB_CE_5.xlsx"

# Parque vehicular (ONASEVI/FONAT, Registro Público de Vehículos). La fuente solo
# publica un tablero Power BI: los dos CSV se extraen con
# src/extraccion/parque_vehicular.py, sin transformar los valores.
PARQUE_FILE = RAW_DIR / "parque_vehicular_onasevi.csv"
PARQUE_CATALOGO_FILE = RAW_DIR / "parque_vehicular_catalogo_onasevi.csv"

# Calendario de fiestas patronales por distrito 2022-2026, mismo formato que
# FERIADOS_FILE (una fila por distrito y día). Compilado con src/extraccion/fiestas.py.
FIESTAS_FILE = RAW_DIR / "fiestas_patronales_elsalvador_2022_2026.xlsx"

# Polígonos OFICIALES de los 262 distritos (Geoportal BCR, capa «Población total
# por distrito» del Censo 2024). No reemplaza a distritos.geojson (capa GADM).
DISTRITOS_GEO_FILE = GEO_DIR / "distritos_censo2024.geojson"

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

    La fila 'No Especificado' (107.055 personas: 94.112 hombres y 12.943 mujeres)
    es la población privada de libertad, que el censo no asigna a ningún distrito.
    Por eso la suma de los 262 distritos (5.922.921) es menor que el total
    nacional (6.029.976).

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


# ---------------------------------------------------------------------------
# 4.1 Cargadores de fuentes externas (etapa final)
# ---------------------------------------------------------------------------
def _quitar_prefijo(serie: pd.Series) -> pd.Series:
    """Quita el prefijo numérico 'NN - ' (o 'NN- ') de los nombres del censo."""
    return serie.astype(str).str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()


def _filas_distrito(df: pd.DataFrame) -> pd.Series:
    """Máscara de filas a nivel distrito en los tabulados del censo (misma regla que load_censo)."""
    return (
        df["distrito"].notna()
        & (~df["distrito"].astype(str).str.contains("TOTAL", na=False))
        & (df["departamento"].astype(str).str.match(r"^\d"))
    )


def load_censo_bienes() -> pd.DataFrame:
    """
    Censo 2024 (BCR/ONEC), tabla TAB_HOG_4: hogares por distrito según
    disponibilidad de automóvil, bicicleta y motocicleta.

    Devuelve 262 distritos con el total de hogares y, por bien, los hogares que
    responden Sí / No / No sabe. Solo normaliza el formato (multi-encabezado,
    prefijos numéricos); los porcentajes se calculan en el notebook.
    Universo: hogares con información completa (1.890.571), según nota del tabulado.
    """
    if not CENSO_BIENES_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {CENSO_BIENES_FILE.name} en {RAW_DIR}."
        )
    df = pd.read_excel(CENSO_BIENES_FILE, header=None, skiprows=6)
    df.columns = [
        "departamento",
        "municipio",
        "distrito",
        "tipo",
        "hogares",
        "automovil_si",
        "automovil_no",
        "automovil_ns",
        "bicicleta_si",
        "bicicleta_no",
        "bicicleta_ns",
        "motocicleta_si",
        "motocicleta_no",
        "motocicleta_ns",
    ]
    d = df[_filas_distrito(df) & (df["tipo"] == "Hogares")].copy()
    num = [
        c
        for c in d.columns
        if c not in ("departamento", "municipio", "distrito", "tipo")
    ]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    for c in ["departamento", "municipio", "distrito"]:
        d[c] = _quitar_prefijo(d[c])
    return d.drop(columns="tipo").reset_index(drop=True)


def load_censo_trabajo() -> pd.DataFrame:
    """
    Censo 2024 (BCR/ONEC), tabla TAB_CE_5: población de 10 años o más ocupada,
    por distrito de residencia, según lugar de trabajo (pregunta P22:
    '1. En este mismo lugar / municipio', '2. En otro lugar', '3. En otro país').

    Devuelve 262 distritos (fila TOTAL de edad), totales por categoría.
    El desglose por sexo y edad queda en el archivo crudo.
    """
    if not CENSO_TRABAJO_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {CENSO_TRABAJO_FILE.name} en {RAW_DIR}."
        )
    df = pd.read_excel(CENSO_TRABAJO_FILE, header=None, skiprows=6).iloc[:, 1:]
    df.columns = [
        "departamento",
        "municipio",
        "distrito",
        "grupo_etario",
        "mismo_lugar_h",
        "mismo_lugar_m",
        "mismo_lugar",
        "otro_lugar_h",
        "otro_lugar_m",
        "otro_lugar",
        "otro_pais_h",
        "otro_pais_m",
        "otro_pais",
        "ns_nr_h",
        "ns_nr_m",
        "ns_nr",
        "ocupados",
    ]
    d = df[_filas_distrito(df) & (df["grupo_etario"] == "TOTAL")].copy()
    cols = ["ocupados", "mismo_lugar", "otro_lugar", "otro_pais", "ns_nr"]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    for c in ["departamento", "municipio", "distrito"]:
        d[c] = _quitar_prefijo(d[c])
    return d[["departamento", "municipio", "distrito", *cols]].reset_index(drop=True)


def load_parque_vehicular() -> pd.DataFrame:
    """
    Parque vehicular (ONASEVI/FONAT) por corte, distrito y clase de vehículo.
    Cortes anuales (31-dic) 2015-2020 y mensuales desde enero de 2021.

    Unifica la columna territorial, que cambió con la reforma municipal:
      - cortes hasta 2024-10-31: el distrito viene en 'cod_mun' (262 antiguos municipios);
      - cortes desde 2024-11-30: viene en 'cod_dis' (mismo espacio de códigos 101-1418).
    En los cortes 2024-09 y 2024-10 'cod_mun' mezcla algunos códigos de municipio
    nuevo; aquí se conservan y su tratamiento se decide en el notebook.
    'cod_distrito' es el código del registro vehicular, no el del censo: el
    cruce con los nombres del censo se hace en el notebook.
    Se excluyen las filas sin código territorial en la fuente (~0,1% del parque).
    """
    if not PARQUE_FILE.exists():
        raise FileNotFoundError(f"No se encontró {PARQUE_FILE.name} en {RAW_DIR}.")
    df = pd.read_csv(PARQUE_FILE, parse_dates=["fecha_ref"])
    nuevo = df["fecha_ref"] >= "2024-11-30"
    df["cod_distrito"] = df["cod_mun"].where(~nuevo, df["cod_dis"]).astype("Int64")
    df["cod_clase"] = df["cod_clase"].astype("Int64")
    return (
        df[["fecha_ref", "cod_distrito", "cod_clase", "vehiculos"]]
        .dropna(subset=["cod_distrito"])
        .reset_index(drop=True)
    )


def load_parque_catalogo() -> pd.DataFrame:
    """Catálogo del registro vehicular: código, distrito y departamento (nombres abreviados del registro)."""
    if not PARQUE_CATALOGO_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {PARQUE_CATALOGO_FILE.name} en {RAW_DIR}."
        )
    return pd.read_csv(PARQUE_CATALOGO_FILE)


def load_fiestas() -> pd.DataFrame:
    """Calendario de fiestas patronales por distrito 2022–2026 (una fila por distrito y día)."""
    if not FIESTAS_FILE.exists():
        raise FileNotFoundError(f"Falta {FIESTAS_FILE.name} en {RAW_DIR}.")
    return pd.read_excel(FIESTAS_FILE)


def load_distritos_geo() -> dict:
    """GeoJSON oficial de los 262 distritos (EPSG:4326), como diccionario.
    No requiere librerías geoespaciales."""
    if not DISTRITOS_GEO_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {DISTRITOS_GEO_FILE.name} en {GEO_DIR}."
        )
    with open(DISTRITOS_GEO_FILE, encoding="utf-8") as f:
        return json.load(f)


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
        "TAB_HOG_4.xlsx (censo, bienes)": CENSO_BIENES_FILE,
        "TAB_CE_5.xlsx (censo, lugar de trabajo)": CENSO_TRABAJO_FILE,
        "parque_vehicular_onasevi.csv": PARQUE_FILE,
        "parque_vehicular_catalogo_onasevi.csv": PARQUE_CATALOGO_FILE,
        "fiestas_patronales_*.xlsx": FIESTAS_FILE,
        "geo/distritos_censo2024.geojson": DISTRITOS_GEO_FILE,
    }
    for nombre, path in base.items():
        print(f"[{'OK ' if path.exists() else 'FALTA'}] {nombre}")

    print()
    for clave, sub in ESTACIONES_CLIMA.items():
        files = _clima_files(CLIMA_DIR / sub)
        estado = f"OK ({len(files)} arch.)" if files else "FALTA"
        print(f"[{estado}] clima/{sub}")
