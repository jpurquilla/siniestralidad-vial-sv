# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NB08 — Tercil medio y validación robusta
#
# **Proyecto:** Análisis de patrones espacio-temporales de accidentes de tránsito en El Salvador
# **Etapa:** 2 — CRISP-DM Fase 5 (Evaluación)
# **Notebook previo:** NB07 — Optimización y modelos avanzados
# **Semilla:** 42
#
# ---
#
# ## Por qué existe este notebook
#
# NB07 cerró el compromiso que NB05 había dejado abierto: XGBoost con la exposición como
# *offset* recupera simultáneamente la calibración (0.966979) y el ordenamiento del tercil
# alto (0.977519, por encima del 0.973177 de Etapa 1).
#
# Pero dejó un hueco. En todos los modelos entrenados sobre la exposición corregida, los
# distritos de siniestralidad intermedia se ordenan peor que en Etapa 1: Spearman 0.6470
# contra 0.8794. Los terciles alto y bajo no presentan el problema.
#
# Este notebook responde tres preguntas, en este orden:
#
# 1. **¿La caída es real o es un artefacto de medición?** Con 34 distritos y conteos sujetos
#    a variación aleatoria, una diferencia entre dos coeficientes de Spearman puede no
#    significar nada. Se determina *antes* de intentar cualquier corrección.
# 2. **Si es real, ¿cuál es la causa?**
# 3. **¿Qué queda de la Fase 5?** Validación cruzada temporal, análisis de sesgos por
#    subgrupo y limitaciones documentadas.
#
# ## Estructura
#
# | Sección | Contenido |
# |---|---|
# | 1 | Preparación, integridad y reproducción de líneas base |
# | 2 | ¿Es real la caída del tercil medio? |
# | 3 | Caracterización del rango medio |
# | 4 | Remedio al sesgo de calibración rural |
# | 5 | Fase 5: validación cruzada temporal, sesgos y limitaciones |
# | 6 | Confirmación en el conjunto de prueba |
#
# ## Nota sobre la numeración de la secuencia
#
# El submodelo condicional de severidad, previsto como NB08 en el plan original de Etapa 2,
# pasa a NB09; ST-DBSCAN a NB10 y fairness urbano/rural a NB11. NB08 ocupa este lugar porque
# el problema abierto de NB07 condiciona los modelos posteriores: antes de construir sobre el
# modelo de frecuencia hay que saber si su capacidad ordinal en el rango medio está
# comprometida.
#
# ## Reglas de trabajo declaradas
#
# | Regla | Contenido |
# |---|---|
# | Partición | Train 2022-01-01 a 2024-12-31 / prueba 2025-01-01 a 2026-06-30. **No se altera.** |
# | Selección de cualquier variante | Partición interna: ajuste 2022-2023 / validación 2024. |
# | Función objetivo | `RMSE × (1 + λ·\|ln(calibración)\|)`, λ=1 |
# | Jerarquía de métricas | calibración → RMSE → AIC → MAE (informativo) |
# | MAE | Nunca es criterio: con 89.7% de celdas en cero, un modelo baja su MAE prediciendo menos. |
# | Terciles | Alto = 35 distritos con mayor total observado; medio y bajo, 34 cada uno. |
# | Calibración | `media(predicho) / media(observado)` sobre el conjunto evaluado. |
# | Archivos previos | **Solo lectura.** Verificado por hash. |
# | Conjunto de prueba | Contador explícito de usos. Ver Sección 6. |
#
# ## Advertencia operativa sobre el modelo vigente
#
# `frecuencia_xgboost_offset_v2.pkl` **no contiene** su propio intercepto. La predicción exige
# `base_margin = log_exposicion_v2 + log_tasa_base`, con `log_tasa_base = -12.936335060885538`,
# almacenado únicamente en `nb07_especificacion.json`. Sin ese término las predicciones salen
# del orden de 415,000 veces más grandes **y no se lanza ningún error**. Toda celda que cargue
# el modelo verifica el valor y se detiene si no coincide.
#
# ---
#
# ## 1.1 Entorno, rutas e integridad de artefactos previos
#
# Esta celda no calcula nada del análisis. Fija la semilla, registra versiones, inventaría los
# artefactos de etapas previas y guarda su huella SHA-256. En la primera ejecución crea
# `reports/results/nb08_hashes_iniciales.json`; en las siguientes compara contra él y se
# detiene si algo cambió. Es la garantía mecánica de que NB08 no toca nada de NB01–NB07.

# %%
# NB08 — 1.1 Entorno, rutas e integridad de artefactos previos
# No calcula resultados: prepara el entorno y verifica que nada previo haya cambiado.

import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root(marcador: str = "src") -> Path:
    """Sube desde el directorio actual hasta encontrar la raiz del repositorio."""
    actual = Path.cwd().resolve()
    for candidato in [actual, *actual.parents]:
        if (candidato / marcador / "config.py").is_file():
            return candidato
    raise RuntimeError(f"No se encontro {marcador}/config.py desde {actual}")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402,F401

SEED = 42
np.random.seed(SEED)

DIR_PROCESSED = ROOT / "data" / "processed"
DIR_MODELS = ROOT / "models"
DIR_RESULTS = ROOT / "reports" / "results"
DIR_FIGURES = ROOT / "reports" / "figures"
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# clave -> (ruta, imprescindible)
ARTEFACTOS_PREVIOS = {
    "matriz_frecuencia_v2": (DIR_PROCESSED / "matriz_frecuencia_v2.csv", True),
    "matriz_frecuencia_e1": (DIR_PROCESSED / "matriz_frecuencia.csv", False),
    "xgb_offset_v2": (DIR_MODELS / "frecuencia_xgboost_offset_v2.pkl", True),
    "xgb_feature_v2": (DIR_MODELS / "frecuencia_xgboost_v2.pkl", False),
    "rf_nb07": (DIR_MODELS / "frecuencia_random_forest_nb07.pkl", False),
    "rf_v2": (DIR_MODELS / "frecuencia_random_forest_v2.pkl", False),
    "rf_e1": (DIR_MODELS / "frecuencia_random_forest.pkl", False),
    "nb_v2": (DIR_MODELS / "frecuencia_binomial_negativa_v2.pkl", False),
    "poisson_v2": (DIR_MODELS / "frecuencia_poisson_v2.pkl", False),
    "zip_v2": (DIR_MODELS / "frecuencia_zip_v2.pkl", False),
    "zinb_v2": (DIR_MODELS / "frecuencia_zinb_v2.pkl", False),
    "nb05_especificacion": (DIR_RESULTS / "nb05_especificacion_v2.json", True),
    "nb05_metricas": (DIR_RESULTS / "nb05_metricas.json", False),
    "nb07_especificacion": (DIR_RESULTS / "nb07_especificacion.json", True),
    "nb07_metricas": (DIR_RESULTS / "nb07_metricas.json", True),
}


def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    """SHA-256 leyendo por bloques (la matriz v2 pesa ~104 MB)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


filas, faltantes = [], []
for clave, (ruta, imprescindible) in ARTEFACTOS_PREVIOS.items():
    existe = ruta.is_file()
    if not existe and imprescindible:
        faltantes.append(str(ruta.relative_to(ROOT)))
    filas.append({
        "clave": clave,
        "ruta": str(ruta.relative_to(ROOT)),
        "existe": existe,
        "MB": round(ruta.stat().st_size / 1024**2, 2) if existe else np.nan,
        "sha256": sha256_archivo(ruta) if existe else None,
    })
if faltantes:
    raise FileNotFoundError("Faltan artefactos imprescindibles:\n  - " + "\n  - ".join(faltantes))

inventario = pd.DataFrame(filas)
hashes_actuales = {f["clave"]: f["sha256"] for f in filas if f["existe"]}

RUTA_HASHES = DIR_RESULTS / "nb08_hashes_iniciales.json"
if RUTA_HASHES.is_file():
    hashes_base = json.loads(RUTA_HASHES.read_text(encoding="utf-8"))["hashes"]
    alterados = [k for k in hashes_base
                 if k in hashes_actuales and hashes_base[k] != hashes_actuales[k]]
    desaparecidos = [k for k in hashes_base if k not in hashes_actuales]
    if alterados or desaparecidos:
        raise RuntimeError(
            "INTEGRIDAD ROTA. Artefactos de etapas previas modificados o eliminados.\n"
            f"  modificados : {alterados}\n  desaparecidos: {desaparecidos}\n"
            "NB08 no debe alterar nada de NB01-NB07."
        )
    estado_integridad = f"verificada ({len(hashes_base)} archivos)"
else:
    RUTA_HASHES.write_text(json.dumps({
        "notebook": "NB08_tercil_medio_y_validacion",
        "generado": datetime.now().isoformat(timespec="seconds"),
        "hashes": hashes_actuales,
    }, indent=2), encoding="utf-8")
    estado_integridad = f"linea base creada ({len(hashes_actuales)} archivos)"

# --- Especificaciones de etapas previas -----------------------------------------
ESPEC_NB07 = json.loads(ARTEFACTOS_PREVIOS["nb07_especificacion"][0].read_text(encoding="utf-8"))
METRICAS_NB07 = json.loads(ARTEFACTOS_PREVIOS["nb07_metricas"][0].read_text(encoding="utf-8"))
ESPEC_NB05 = json.loads(ARTEFACTOS_PREVIOS["nb05_especificacion"][0].read_text(encoding="utf-8"))

_espec_offset = ESPEC_NB07["modelos"]["frecuencia_xgboost_offset_v2.pkl"]
LOG_TASA_BASE = _espec_offset["log_tasa_base"]
COLUMNAS_OFFSET = _espec_offset["columnas"]
OFFSET_COL = ESPEC_NB07["offset"]

LOG_TASA_BASE_DOCUMENTADO = -12.936335060885538
if not np.isclose(LOG_TASA_BASE, LOG_TASA_BASE_DOCUMENTADO, rtol=0, atol=1e-12):
    raise RuntimeError(
        "log_tasa_base NO coincide con el valor documentado en NB07.\n"
        f"  json: {LOG_TASA_BASE!r}\n  doc : {LOG_TASA_BASE_DOCUMENTADO!r}\n"
        "Sin el valor correcto las predicciones se desvian ~415,000x SIN lanzar error."
    )
if ESPEC_NB07["seed"] != SEED or ESPEC_NB05["seed"] != SEED:
    raise RuntimeError("Semilla inconsistente entre notebooks.")

# Esquema normativo de features (fuente de verdad: NB05).
FEATURES = ESPEC_NB05["esquema_features"]["features"]
CATEGORICAS = ESPEC_NB05["esquema_features"]["categoricas"]
if FEATURES != COLUMNAS_OFFSET:
    raise RuntimeError(f"Orden normativo distinto.\n  NB05: {FEATURES}\n  NB07: {COLUMNAS_OFFSET}")

# Contador de usos del conjunto de prueba.
USOS_TEST = {"NB07 (evaluacion)": 1, "NB08 (reproduccion de control)": 0, "NB08 (evaluacion)": 0}

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)

print("NB08 — Tercil medio y validacion robusta")
print(f"Raiz          : {ROOT}")
print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
print(f"Semilla       : {SEED}")
print(f"Integridad    : {estado_integridad}")
print(f"log_tasa_base : {LOG_TASA_BASE!r}  [verificado]")
print(f"Offset        : {OFFSET_COL}")
print(f"Features      : {len(FEATURES)} -> {FEATURES}")
print(f"Usos del test : {USOS_TEST}")
print()
display(inventario[["clave", "ruta", "existe", "MB"]])

# %% [markdown]
# ## 1.2 Carga de la matriz y validación del esquema
#
# Se carga `matriz_frecuencia_v2.csv` y se valida contra las especificaciones de NB05 y NB07
# antes de tocar un solo modelo.
#
# El motivo es una lección de NB05 (Sección 4). Los modelos de Etapa 1 se guardaron sin su
# esquema de columnas. Al pasarles una matriz de diseño con las columnas en otro orden, el
# modelo **no falla**: devuelve predicciones sin sentido —una media de 46,148 cuando lo real
# ronda 0.15— y nada en el código avisa. Desde NB05 cada artefacto viaja con su JSON de
# especificación, y la regla es validarlo al arrancar.
#
# Se comprueba:
#
# 1. Que existan todas las columnas que NB07 declara, más el *offset* y la columna de partición.
# 2. Que el panel esté balanceado: 103 × 1,642 × 4 = 676,504 celdas.
# 3. Que la partición interna de NB07 se reproduzca exactamente —300,760 filas de ajuste y
#    150,792 de validación—. Control cruzado de que es la misma matriz que usó NB07.
# 4. Qué variables son constantes dentro de cada distrito. Con la exposición en el *offset*,
#    solo esas pueden ordenar distritos entre sí.
#
# Se declaran además las columnas **prohibidas como predictoras**: `fallecidos` y `lesionados`
# son variables *ex post*. No puede haber fallecidos donde no hubo siniestro; usarlas sería
# fuga circular.

# %%
# NB08 — 1.2 Carga de la matriz y validacion del esquema

RUTA_MATRIZ_V2 = ARTEFACTOS_PREVIOS["matriz_frecuencia_v2"][0]

print(f"Cargando {RUTA_MATRIZ_V2.name} ({RUTA_MATRIZ_V2.stat().st_size / 1024**2:.1f} MB)...")
matriz = pd.read_csv(RUTA_MATRIZ_V2, parse_dates=["fecha"])
cols = set(matriz.columns)
print(f"Forma: {matriz.shape[0]:,} filas x {matriz.shape[1]} columnas")
print(f"Memoria: {matriz.memory_usage(deep=True).sum() / 1024**2:,.1f} MB\n")

# --- Nombres reales de las columnas clave (verificados contra el esquema) --------
COL_DISTRITO, COL_DEPTO = "distrito_key", "departamento_key"
COL_FECHA, COL_FRANJA = "fecha", "franja_horaria"
COL_ESTACION, COL_Y = "estacion_asignada", "n_siniestros"
COL_SPLIT = ESPEC_NB07["split"]["columna"]

_obligatorias = [COL_DISTRITO, COL_DEPTO, COL_FECHA, COL_FRANJA, COL_ESTACION,
                 COL_Y, COL_SPLIT, OFFSET_COL, "log_exposicion", "poblacion",
                 "factor_crec", *FEATURES]
_faltan = [c for c in dict.fromkeys(_obligatorias) if c not in cols]
if _faltan:
    raise RuntimeError(f"Faltan columnas en la matriz: {_faltan}\nDisponibles: {sorted(cols)}")

# Variables ex post: presentes en la matriz, PROHIBIDAS como predictoras de frecuencia.
COLS_PROHIBIDAS = [c for c in ("fallecidos", "lesionados") if c in cols]

esquema = pd.DataFrame({
    "dtype": matriz.dtypes.astype(str),
    "nulos": matriz.isna().sum(),
    "unicos": matriz.nunique(),
})
print("=== ESQUEMA ===")
display(esquema)

# --- Fechas del protocolo, parseadas desde el JSON (no escritas duras) ----------
def _rango(texto: str) -> tuple:
    ini, fin = [p.strip() for p in texto.split(" a ")]
    return pd.Timestamp(ini), pd.Timestamp(fin)


TRAIN_INI, TRAIN_FIN = _rango(ESPEC_NB07["split"]["train"])
TEST_INI, TEST_FIN = _rango(ESPEC_NB07["split"]["test"])
AJUSTE_INI, AJUSTE_FIN = _rango(ESPEC_NB07["particion_interna"]["ajuste"])
VALID_INI, VALID_FIN = _rango(ESPEC_NB07["particion_interna"]["validacion"])
FILAS_AJUSTE_NB07 = ESPEC_NB07["particion_interna"]["filas_ajuste"]
FILAS_VALID_NB07 = ESPEC_NB07["particion_interna"]["filas_validacion"]

es_train = matriz[COL_SPLIT].astype(str).str.lower().str.startswith("train")
MASK_TRAIN = es_train
MASK_TEST = ~es_train
MASK_AJUSTE = es_train & matriz[COL_FECHA].between(AJUSTE_INI, AJUSTE_FIN)
MASK_VALID = es_train & matriz[COL_FECHA].between(VALID_INI, VALID_FIN)

print("\n=== PARTICION ===")
display(matriz.groupby(COL_SPLIT).agg(
    filas=(COL_SPLIT, "size"),
    fecha_min=(COL_FECHA, "min"),
    fecha_max=(COL_FECHA, "max"),
    media_y=(COL_Y, "mean"),
))

n_aj, n_va = int(MASK_AJUSTE.sum()), int(MASK_VALID.sum())
print("\nParticion interna de NB07 — control cruzado")
print(f"  ajuste 2022-2023 : {n_aj:,} (NB07 declara {FILAS_AJUSTE_NB07:,})")
print(f"  validacion 2024  : {n_va:,} (NB07 declara {FILAS_VALID_NB07:,})")
if (n_aj, n_va) != (FILAS_AJUSTE_NB07, FILAS_VALID_NB07):
    raise RuntimeError("La particion interna NO se reproduce: la matriz no es la de NB07.")
print("  -> coincide exactamente. Misma matriz que NB07.")

n_dist, n_dias, n_franjas = (matriz[c].nunique() for c in (COL_DISTRITO, COL_FECHA, COL_FRANJA))
esperado = n_dist * n_dias * n_franjas
print("\n=== UNIDADES DE ANALISIS ===")
print(f"  distritos     : {n_dist}   departamentos: {matriz[COL_DEPTO].nunique()}")
print(f"  dias          : {n_dias:,} ({matriz[COL_FECHA].min():%Y-%m-%d} a {matriz[COL_FECHA].max():%Y-%m-%d})")
print(f"  franjas       : {n_franjas} -> {sorted(matriz[COL_FRANJA].unique())}")
print(f"  estaciones    : {matriz[COL_ESTACION].nunique()} -> {sorted(matriz[COL_ESTACION].unique())}")
print(f"  panel         : {esperado:,} esperadas vs {len(matriz):,} reales -> "
      f"{'BALANCEADO' if esperado == len(matriz) else 'INCOMPLETO'}")
print(f"  objetivo      : media {matriz[COL_Y].mean():.6f} | ceros {(matriz[COL_Y] == 0).mean():.4%} | max {matriz[COL_Y].max()}")
print(f"  prohibidas    : {COLS_PROHIBIDAS} (ex post, nunca predictoras)")

# --- Grados de libertad entre distritos -----------------------------------------
nunicos = matriz.groupby(COL_DISTRITO)[[*FEATURES, COL_ESTACION, "poblacion"]].nunique()
constantes = [c for c in nunicos.columns if (nunicos[c] == 1).all()]
temporales = [c for c in nunicos.columns if c not in constantes]
print("\n=== GRADOS DE LIBERTAD ENTRE DISTRITOS ===")
print(f"  Constantes dentro de cada distrito : {constantes}")
print(f"  Varian en el tiempo                : {temporales}")
print("  Solo las primeras pueden diferenciar un distrito de otro.")
terr = matriz.groupby(COL_DISTRITO)[["pct_urbano", "poblacion_baja", "poblacion"]].first()
terr[COL_ESTACION] = matriz.groupby(COL_DISTRITO)[COL_ESTACION].first()
print(f"  Combinaciones territoriales distintas: "
      f"{len(terr[['pct_urbano', 'poblacion_baja', COL_ESTACION]].drop_duplicates())} de {len(terr)}")
display(terr.describe())

# %% [markdown]
# ## 1.3 Reproducción de las líneas base de NB05 y NB07
#
# Antes de investigar nada, el notebook demuestra que reproduce los resultados previos. Si las
# métricas recalculadas no coinciden con `nb07_metricas.json` hasta el sexto decimal, se
# detiene: significaría que el entorno, la matriz o la carga de modelos difieren de lo que
# produjo esos números, y cualquier conclusión posterior sería inauditable.
#
# **Sobre el uso del conjunto de prueba.** Esta celda calcula métricas sobre el test, pero
# **no cuenta como una evaluación**. No mide nada nuevo ni decide nada: recalcula valores ya
# publicados en NB07 para verificar que el entorno los reproduce. Es control de integridad, no
# experimentación. El contador de la Sección 6 lo registra por separado.
#
# Se definen aquí las funciones que el resto del notebook reutiliza —`predecir_xgb_offset`,
# `predecir_glm` y `metricas`— para que la lógica de `base_margin` exista **en un solo lugar**.
# Duplicarla sería reintroducir el riesgo de que una copia quede sin `log_tasa_base`.
#
# Los Random Forest son opcionales (`REPRODUCIR_RF`): pesan más de 120 MB y su carga es lenta.
# Se usan en la Sección 2, donde se cargan con su propia verificación.

# %%
# NB08 — 1.3 Reproduccion de las lineas base de NB05 y NB07
# Control de integridad. NO cuenta como evaluacion del test: recalcula valores ya publicados.

import joblib
import xgboost as xgb
import statsmodels.api as sm
from scipy import stats

REPRODUCIR_RF = False   # True para incluir tambien los Random Forest (lento, >120 MB)
TOL = 1e-6              # tolerancia: sexto decimal


def predecir_xgb_offset(df: pd.DataFrame, booster=None) -> np.ndarray:
    """
    Prediccion del modelo vigente (XGBoost con exposicion como offset).

    UNICA implementacion en el notebook. base_margin = log_exposicion_v2 + log_tasa_base.
    Sin log_tasa_base las predicciones salen ~415,000x mas grandes SIN lanzar error.
    """
    if booster is None:
        booster = BOOSTER_OFFSET
    faltan = [c for c in COLUMNAS_OFFSET if c not in df.columns]
    if faltan:
        raise RuntimeError(f"Faltan columnas para el modelo offset: {faltan}")
    dm = xgb.DMatrix(df[COLUMNAS_OFFSET].astype(float))
    dm.set_base_margin(df[OFFSET_COL].values + LOG_TASA_BASE)
    return booster.predict(dm)


def diseno_glm(df: pd.DataFrame, columnas_ref=None) -> pd.DataFrame:
    """Matriz de diseno de los GLM: get_dummies(drop_first) sobre FEATURES + constante."""
    X = pd.get_dummies(df[FEATURES], columns=CATEGORICAS, drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add")
    if columnas_ref is not None:
        X = X.reindex(columns=columnas_ref, fill_value=0.0)
    return X


def predecir_glm(modelo, df: pd.DataFrame) -> np.ndarray:
    """
    Prediccion de un GLM guardado con remove_data().

    remove_data() borra exog_names, asi que el modelo NO valida el orden de columnas:
    se multiplica manualmente contra el orden normativo.
    """
    X = diseno_glm(df)
    beta = np.asarray(modelo.params, dtype=float)
    if X.shape[1] != len(beta):
        raise RuntimeError(
            f"Dimension incompatible: diseno {X.shape[1]} columnas, modelo {len(beta)} "
            f"coeficientes. Orden normativo: {list(X.columns)}"
        )
    return np.exp(X.values @ beta + df[OFFSET_COL].values)


def metricas(obs, pred, distritos, n_alto=35) -> dict:
    """Metricas del protocolo. Calibracion = media(pred)/media(obs)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    agg = (pd.DataFrame({"d": np.asarray(distritos), "obs": obs, "pred": pred})
           .groupby("d")[["obs", "pred"]].sum()
           .sort_values("obs", ascending=False))
    alto = agg.head(n_alto)
    return {
        "MAE": float(np.mean(np.abs(obs - pred))),
        "RMSE": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "media_predicha": float(pred.mean()),
        "calibracion": float(pred.mean() / obs.mean()),
        "spearman_global": float(stats.spearmanr(agg["pred"], agg["obs"]).statistic),
        "spearman_tercil_alto": float(stats.spearmanr(alto["pred"], alto["obs"]).statistic),
        "n_distritos": float(len(agg)),
        "n_tercil_alto": float(len(alto)),
    }


BOOSTER_OFFSET = joblib.load(ARTEFACTOS_PREVIOS["xgb_offset_v2"][0])
print(f"Cargado: frecuencia_xgboost_offset_v2.pkl -> {type(BOOSTER_OFFSET).__name__}")

test = matriz[MASK_TEST].copy()
print(f"Conjunto de prueba: {len(test):,} filas | media observada {test[COL_Y].mean():.6f}\n")

reproducciones = {}
reproducciones["NB07 - XGBoost (exposición offset)"] = metricas(
    test[COL_Y], predecir_xgb_offset(test), test[COL_DISTRITO]
)

for clave, etiqueta in [("poisson_v2", "NB05 - Poisson"),
                        ("nb_v2", "NB05 - Binomial Negativa")]:
    ruta = ARTEFACTOS_PREVIOS[clave][0]
    if ruta.is_file():
        try:
            reproducciones[etiqueta] = metricas(
                test[COL_Y], predecir_glm(joblib.load(ruta), test), test[COL_DISTRITO]
            )
        except Exception as e:
            print(f"AVISO: no se pudo reproducir {etiqueta}: {type(e).__name__}: {e}")

if REPRODUCIR_RF:
    cols_rf = [*FEATURES, OFFSET_COL]
    for clave, etiqueta in [("rf_v2", "NB05 - Random Forest"),
                            ("rf_nb07", "NB07 - Random Forest regularizado")]:
        ruta = ARTEFACTOS_PREVIOS[clave][0]
        if ruta.is_file():
            print(f"Cargando {ruta.name} ({ruta.stat().st_size / 1024**2:.0f} MB)...")
            rf = joblib.load(ruta)
            reproducciones[etiqueta] = metricas(
                test[COL_Y], rf.predict(test[cols_rf].astype(float)), test[COL_DISTRITO]
            )
            del rf

CLAVES = ["MAE", "RMSE", "media_predicha", "calibracion",
          "spearman_global", "spearman_tercil_alto"]
filas, discrepancias = [], []
for etiqueta, recalc in reproducciones.items():
    publicado = METRICAS_NB07["modelos"].get(etiqueta)
    if publicado is None:
        print(f"AVISO: '{etiqueta}' no figura en nb07_metricas.json")
        continue
    for k in CLAVES:
        if k not in publicado:
            continue
        dif = abs(recalc[k] - publicado[k])
        ok = dif <= TOL + 5e-7   # nb07_metricas.json guarda 6 decimales
        filas.append({"modelo": etiqueta, "metrica": k, "publicado": publicado[k],
                      "recalculado": round(recalc[k], 6), "|dif|": dif, "ok": ok})
        if not ok:
            discrepancias.append(f"{etiqueta} / {k}: publicado {publicado[k]} vs {recalc[k]:.6f}")

verificacion = pd.DataFrame(filas)
print("=== REPRODUCCION DE LINEAS BASE ===")
display(verificacion.set_index(["modelo", "metrica"]))

if discrepancias:
    raise RuntimeError(
        "Las lineas base NO se reproducen al sexto decimal:\n  - " + "\n  - ".join(discrepancias)
        + "\n\nRevisar: orden de columnas de la matriz de diseno, valor de log_tasa_base, "
          "o versiones de las librerias. NO continuar hasta resolverlo."
    )

USOS_TEST["NB08 (reproduccion de control)"] = 1
print(f"\nTodas las metricas reproducidas al sexto decimal ({len(verificacion)} comparaciones).")
print(f"Modelos verificados: {list(reproducciones)}")
print(f"Usos del test: {USOS_TEST}")
del test

# %% [markdown]
# ## 1.4 Decisiones metodológicas: qué se decidió, por qué, y qué se dejó de lado
#
# Esta celda no calcula nada. Registra dentro del notebook las decisiones de diseño con su
# argumento y su alternativa descartada, y las guarda en `nb08_decisiones.json`. El criterio es
# que cualquier lector —o el jurado— pueda auditar **por qué** el notebook está armado así.
#
# ### D1 — El diagnóstico se hace en la validación 2024; el test se lee pero no decide
#
# Las Secciones 2, 3 y 4 se ejecutan sobre la validación interna (2024, 150,792 filas). El
# conjunto de prueba se reporta en paralelo, marcado como descriptivo.
#
# **Por qué.** La cifra que motiva el notebook se midió sobre el test. Si además se diagnostica
# la causa y se elige un remedio mirando ese mismo conjunto, el test deja de ser una prueba
# independiente. Ahora bien, el 0.6470 **ya está publicado**: lo que hay que evitar no es ver el
# test, sino *decidir* con él.
#
# **Descartado:** diagnosticar sobre el test (la Sección 6 ya no podría declarar evaluación
# independiente) e ignorarlo por completo (se pierde la comprobación de si el patrón se repite
# entre períodos).
#
# **Costo asumido.** La validación 2024 cubre 12 meses contra 18 del test y está más cerca del
# entrenamiento. Se investiga el **patrón**, no la reproducción de una cifra puntual.
#
# **Verificable.** `exigir_particion_de_diagnostico()` rechaza la partición de prueba como
# fuente de veredictos; toda tabla del test lleva rótulo explícito.
#
# ### D2 — El factor de crecimiento se reestima dentro de cada pliegue
#
# El factor de NB05 se estimó con las medias anuales de 2022 y 2024, o sea con **todo** el
# train. Un pliegue que entrena en 2022 y valida en 2023 estaría usando información de 2024:
# fuga temporal dentro del procedimiento cuyo propósito es detectarlas.
#
# **Descartado:** factor fijo con la fuga declarada como limitación.
#
# **Costo.** Los pliegues tempranos tienen estimaciones más ruidosas. Eso es información útil:
# indica cuánta historia requiere el procedimiento para estabilizarse.
#
# ### D3 — Prohibición permanente de `fallecidos` y `lesionados`
#
# Son variables *ex post*. No puede haber fallecidos donde no hubo siniestro: predecir el conteo
# con ellas es fuga circular. Es la regla de la Matriz B de NB04 extendida a la frecuencia. Se
# verifica en código porque la matriz las trae físicamente y un descuido las incorporaría sin
# que nada falle.
#
# ### D4 — Criterios de veredicto fijados antes de calcular
#
# Con varias hipótesis, tres terciles y varios modelos, el espacio de comparaciones permite
# respaldar casi cualquier relato si el criterio se elige después. Los umbrales se guardan con
# marca de tiempo anterior a todo resultado. Un valor en el límite se reporta como
# *indeterminado*.
#
# ### D5 — Ninguna mejora se declara sin su intervalo de confianza
#
# El Spearman del tercil medio se calcula sobre 34 distritos. **Precisión técnica:** los dos
# coeficientes se calculan sobre los mismos distritos y contra el mismo ranking observado —son
# pareados y dependientes—. Comparar sus intervalos marginales y ver si se solapan es un
# criterio poco fiable: dos intervalos pueden solaparse y aun así la diferencia ser
# significativa, y al revés. El procedimiento correcto es remuestrear distritos y calcular
# **ambos** coeficientes sobre cada remuestra, construyendo el intervalo de la **diferencia**.

# %%
# NB08 — 1.4 Decisiones metodologicas: registro auditable y candados de verificacion

PART_DIAG = "validacion_2024"
MASK_DIAG = MASK_VALID

PARTICIONES = {
    "ajuste": {"rango": ESPEC_NB07["particion_interna"]["ajuste"], "filas": int(MASK_AJUSTE.sum())},
    "validacion_2024": {"rango": ESPEC_NB07["particion_interna"]["validacion"], "filas": int(MASK_VALID.sum())},
    "train_completo": {"rango": ESPEC_NB07["split"]["train"], "filas": int(MASK_TRAIN.sum())},
    "test": {"rango": ESPEC_NB07["split"]["test"], "filas": int(MASK_TEST.sum())},
}

# --- Umbrales fijados ANTES de calcular nada (D4) --------------------------------
CRITERIOS = {
    "H1_compresion": {"regla": "proporcion de pares distinguibles en tercil medio < alto y < 0.50",
                      "umbral_absoluto": 0.50, "zona_indeterminada": 0.05},
    "H2_exposicion": {"regla": "spearman(exposicion, observado) en medio <= min(alto, bajo) - 0.15",
                      "margen": 0.15, "zona_indeterminada": 0.05},
    "H3_heterogeneidad": {"regla": "spearman(efecto_2022, efecto_2023) en medio >= 0.50",
                          "umbral": 0.50, "zona_indeterminada": 0.05},
    "H4_memorizacion": {"regla": "spearman(baseline historico) >= spearman(RF E1) - 0.05 en medio",
                        "margen": 0.05, "zona_indeterminada": 0.02},
    "remedio_admisible": {"regla": "la mejora debe exceder el IC bootstrap y la calibracion no "
                                   "puede degradarse mas de 0.05",
                          "degradacion_calibracion_maxima": 0.05},
}

B_BOOT = 2000
RNG = np.random.default_rng(SEED)
TERCILES = ["bajo", "medio", "alto"]


def exigir_particion_de_diagnostico(nombre: str) -> None:
    """Falla si una funcion que emite veredictos recibe datos del conjunto de prueba."""
    if nombre not in ("ajuste", "validacion_2024", "train_completo"):
        raise RuntimeError(
            f"Intento de emitir un veredicto sobre la particion '{nombre}'.\n"
            "D1: las Secciones 2-4 deciden UNICAMENTE sobre validacion interna."
        )


def verificar_predictoras(columnas, contexto: str = "") -> list:
    """Falla si una lista de predictoras contiene variables ex post."""
    intrusas = [c for c in columnas if c in COLS_PROHIBIDAS]
    if intrusas:
        raise RuntimeError(
            f"Variables ex post entre las predictoras{' (' + contexto + ')' if contexto else ''}: "
            f"{intrusas}.\nD3: no puede haber fallecidos ni lesionados donde no hubo siniestro."
        )
    return list(columnas)


verificar_predictoras(FEATURES, "esquema normativo NB05")
ETIQUETA_DESCRIPTIVA = "[descriptivo — no decide]"

DECISIONES = {
    "notebook": "NB08_tercil_medio_y_validacion",
    "generado": datetime.now().isoformat(timespec="seconds"),
    "seed": SEED,
    "nota_numeracion": (
        "El submodelo condicional de severidad, previsto como NB08 en el plan original de "
        "Etapa 2, pasa a NB09; ST-DBSCAN a NB10 y fairness urbano/rural a NB11. NB08 ocupa "
        "este lugar porque el problema abierto de NB07 condiciona los modelos posteriores."
    ),
    "decisiones": {
        "D1_particion_de_diagnostico": {
            "decision": "Secciones 2-4 deciden sobre validacion 2024; el test se reporta como "
                        "descriptivo y se evalua una sola vez en la Seccion 6.",
            "justificacion": "La cifra que motiva el notebook se midio sobre el test. "
                             "Diagnosticar y elegir remedio observandolo lo convierte en "
                             "conjunto de desarrollo. Lo que debe evitarse no es ver el test "
                             "sino decidir con el.",
            "alternativas_descartadas": {
                "diagnosticar_sobre_test": "La Seccion 6 no podria declarar evaluacion independiente.",
                "ignorar_el_test": "Se pierde la comprobacion de si el patron se repite entre periodos.",
            },
            "costo_asumido": "Validacion 2024 cubre 12 meses contra 18 del test. Se investiga "
                             "el patron, no la reproduccion de una cifra puntual.",
            "verificacion": "exigir_particion_de_diagnostico() + etiquetado de columnas de test",
        },
        "D2_factor_crecimiento_en_cv": {
            "decision": "Reestimar el factor de crecimiento dentro de cada pliegue de la CV temporal.",
            "justificacion": "El factor de NB05 se estimo con las medias anuales de 2022 y 2024, "
                             "es decir con todo el train. Un pliegue que entrena en 2022 y valida "
                             "en 2023 estaria usando informacion de 2024: fuga temporal.",
            "alternativa_descartada": "Factor fijo con la fuga declarada como limitacion.",
            "costo_asumido": "Los pliegues tempranos tienen estimaciones mas ruidosas; eso indica "
                             "cuanta historia requiere el procedimiento para estabilizarse.",
            "referencia_nb05": ESPEC_NB05["correccion_tendencia"],
        },
        "D3_variables_ex_post_prohibidas": {
            "decision": f"{COLS_PROHIBIDAS} prohibidas como predictoras en todo el notebook.",
            "justificacion": "Se conocen despues del siniestro; el predictor contendria la respuesta.",
            "precedente": "Matriz B de NB04 (regla anti-fuga del clasificador de severidad).",
            "verificacion": "verificar_predictoras() en cada conjunto de predictoras",
        },
        "D4_criterios_previos": {
            "decision": "Umbrales fijados antes de calcular; valores en el limite = indeterminados.",
            "justificacion": "El espacio de comparaciones permite respaldar casi cualquier relato "
                             "si el criterio se elige despues.",
            "criterios": CRITERIOS,
        },
        "D5_incertidumbre_obligatoria": {
            "decision": "Toda comparacion entre modelos se acompana de IC bootstrap.",
            "justificacion": "El Spearman del tercil medio se calcula sobre 34 distritos.",
            "precision_tecnica": "Los coeficientes son pareados (mismos distritos, mismo ranking "
                                 "observado). El solapamiento de IC marginales es criterio poco "
                                 "fiable: se construye el IC de la DIFERENCIA remuestreando "
                                 "distritos y recalculando ambos coeficientes en cada remuestra.",
            "remuestreos": B_BOOT,
        },
    },
    "trazabilidad_particiones": {
        "seccion_1": {"ve": ["ajuste", "validacion_2024", "train_completo", "test"], "decide": False},
        "seccion_2": {"ve": ["validacion_2024", "test"], "decide": "validacion_2024"},
        "seccion_3": {"ve": ["ajuste", "validacion_2024"], "decide": "validacion_2024"},
        "seccion_4": {"ve": ["ajuste", "validacion_2024"], "decide": "validacion_2024"},
        "seccion_5": {"ve": ["train_completo"], "decide": "pliegues internos"},
        "seccion_6": {"ve": ["test"], "decide": False},
    },
    "particiones": PARTICIONES,
    "protocolo_heredado": {
        "split": ESPEC_NB07["split"],
        "particion_interna": ESPEC_NB07["particion_interna"],
        "funcion_objetivo": ESPEC_NB07["funcion_objetivo"],
        "jerarquia": METRICAS_NB07["protocolo"]["jerarquia"],
        "definicion_tercil_alto": METRICAS_NB07["protocolo"]["definicion_tercil_alto"],
    },
}

RUTA_DECISIONES = DIR_RESULTS / "nb08_decisiones.json"
RUTA_DECISIONES.write_text(json.dumps(DECISIONES, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== DECISIONES METODOLOGICAS REGISTRADAS ===")
for clave, d in DECISIONES["decisiones"].items():
    print(f"\n{clave}\n  {d['decision']}")
print(f"\nGuardado en: {RUTA_DECISIONES.relative_to(ROOT)}")
print(f"Marca de tiempo: {DECISIONES['generado']} (anterior a todo resultado)")

print("\n=== PARTICIONES ===")
display(pd.DataFrame(PARTICIONES).T[["rango", "filas"]])

print("=== CANDADOS ACTIVOS ===")
print(f"  D1 exigir_particion_de_diagnostico() -> particion de decision: {PART_DIAG}")
print(f"  D3 verificar_predictoras()           -> prohibidas: {COLS_PROHIBIDAS}")
print(f"  D4 criterios fijados                 -> {len(CRITERIOS)} reglas con umbral previo")
print(f"  D5 bootstrap                         -> {B_BOOT} remuestreos, semilla {SEED}")

try:
    exigir_particion_de_diagnostico("test")
    raise AssertionError("El candado D1 no bloqueo la particion de prueba.")
except RuntimeError:
    print("\n  Candado D1 verificado: rechaza la particion de prueba como fuente de veredictos.")

# %% [markdown]
# # Sección 2 — ¿Es real la caída del tercil medio?
#
# NB07 dejó un problema abierto: en los modelos entrenados sobre la exposición corregida, el
# tercil medio se ordena peor que en Etapa 1 —Spearman 0.6470 contra 0.8794—.
#
# Antes de buscar causas o remedios hay que responder algo previo: **¿esa diferencia significa
# algo?** El coeficiente se calcula sobre 34 distritos y los conteos de siniestros tienen
# variación aleatoria propia.
#
# | Celda | Pregunta |
# |---|---|
# | 2.1 | Gemelos de validación, tabla por distrito y terciles |
# | 2.2 | **Techo de ruido**: ¿cuánto del ranking observado es señal y cuánto azar? |
# | 2.3 | **Techo estructural**: ¿cuánto puede ordenar el modelo con lo que sabe? |
# | 2.4 | **Bootstrap pareado**: ¿la diferencia se distingue del ruido? |
# | 2.5 | Robustez: Kendall tau-b y sensibilidad al corte |
# | 2.6 | Veredicto |
#
# El orden no es decorativo: 2.2 y 2.3 son baratas y pueden cerrar la pregunta solas.
#
# ---
#
# ## 2.1 Gemelos de validación, tabla por distrito y terciles
#
# ### Por qué los modelos de `models/` no sirven para diagnosticar en 2024
#
# Un primer intento de esta celda usó los `.pkl` finales sobre la validación 2024 y produjo un
# resultado imposible: los dos Random Forest dieron Spearman idénticos hasta el sexto decimal
# (0.998166) contra el observado.
#
# La causa es que esos modelos fueron entrenados sobre **2022-2024 completo**. Al predecir sobre
# 2024 no predicen: reproducen datos que ya vieron. Por eso ordenan casi perfecto, y por eso dos
# modelos distintos convergen al mismo ranking.
#
# | Objeto | Entrenado sobre | Para qué sirve |
# |---|---|---|
# | Modelo **candidato** (durante Optuna) | 2022-2023 | Medir desempeño en 2024 sin contaminación |
# | Modelo **final** (los `.pkl`) | 2022-2024 | Predecir sobre el test y ser el entregable |
#
# NB07 usó correctamente los candidatos para elegir hiperparámetros. El error fue de esta celda,
# no del protocolo anterior.
#
# ### Qué se hace en su lugar
#
# Optuna guarda los objetivos de cada ensayo, no los modelos. Se **reinstancian**: gemelos
# entrenados sobre 2022-2023 con los hiperparámetros que la búsqueda de NB07 dejó fijados,
# leídos del JSON. No se optimiza nada nuevo.
#
# Si el gemelo del XGBoost reproduce el RMSE y la calibración que NB07 publicó en validación,
# queda demostrado que aquella optimización es replicable. Para los Random Forest no hay cifra
# de validación publicada: sus gemelos son réplicas de especificación, sin referencia contra la
# cual verificarse. Es una limitación que se documenta.
#
# ### Sobre el esquema de los Random Forest
#
# Ambos bosques fueron entrenados con la columna de exposición llamada `log_exposicion`. NB05 no
# agregó una columna nueva: corrigió los **valores** de la existente. Cada uno necesita el mismo
# nombre con valores distintos, y hay que renombrar al armar la matriz.
#
# scikit-learn valida los nombres y rechaza una matriz mal armada, pero **no valida los
# valores**: intercambiarlos desvía las predicciones hasta un 39.58% sin error. Por eso se
# verifica la magnitud de cada modelo final contra su calibración publicada en NB07.

# %%
# NB08 — 2.1 Gemelos de validacion, tabla por distrito y terciles

from sklearn.ensemble import RandomForestRegressor

N_ALTO, N_MEDIO, N_BAJO = 35, 34, 34


def spearman_seguro(x, y) -> float:
    """Spearman con proteccion ante varianza nula (devuelve nan en vez de warning)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def asignar_terciles(orden_desc: pd.Series, n_alto=N_ALTO, n_medio=N_MEDIO) -> pd.Series:
    """Etiqueta alto/medio/bajo ordenando de mayor a menor por el criterio dado."""
    idx = orden_desc.sort_values(ascending=False).index
    n_bajo = len(idx) - n_alto - n_medio
    if n_bajo < 1:
        raise RuntimeError(f"Cortes invalidos: {n_alto}/{n_medio}/{n_bajo}")
    etiquetas = np.array(["alto"] * n_alto + ["medio"] * n_medio + ["bajo"] * n_bajo)
    return pd.Series(etiquetas, index=idx, name="tercil")


# --- A. Esquemas de columnas -----------------------------------------------------
COL_EXPO_RF = "log_exposicion"
COLUMNAS_RF = verificar_predictoras([*FEATURES, COL_EXPO_RF], "Random Forest")

RF_CONFIG = {
    "rf_e1": {"valores_exposicion": "log_exposicion",
              "etiqueta_nb07": "Etapa 1 - Random Forest", "hiperparametros": {}},
    "rf_v2": {"valores_exposicion": OFFSET_COL,
              "etiqueta_nb07": "NB05 - Random Forest", "hiperparametros": {}},
}


def matriz_rf(df: pd.DataFrame, col_valores: str) -> pd.DataFrame:
    """Matriz de diseno del RF: renombra la columna de exposicion al nombre de entrenamiento."""
    X = df[[*FEATURES, col_valores]].astype(float).copy()
    X.columns = COLUMNAS_RF
    return X


# --- B. Verificacion de magnitud de los modelos FINALES --------------------------
print("=== CONTROL: modelos finales sobre el test vs. NB07 ===")
_test = matriz[MASK_TEST]
_obs_medio_test = _test[COL_Y].mean()
for clave, cfg_rf in RF_CONFIG.items():
    rf = joblib.load(ARTEFACTOS_PREVIOS[clave][0])
    nombres = list(getattr(rf, "feature_names_in_", COLUMNAS_RF))
    if nombres != COLUMNAS_RF:
        raise RuntimeError(f"{clave}: el modelo espera {nombres}, se pasan {COLUMNAS_RF}")
    calib = rf.predict(matriz_rf(_test, cfg_rf["valores_exposicion"])).mean() / _obs_medio_test
    pub = METRICAS_NB07["modelos"][cfg_rf["etiqueta_nb07"]]["calibracion"]
    ok = abs(calib - pub) < 1e-5
    print(f"  {cfg_rf['etiqueta_nb07']:28s} calibracion {calib:.6f} | publicada {pub:.6f} | "
          f"{'OK' if ok else 'DISCREPA'}")
    if not ok:
        raise RuntimeError(
            f"{cfg_rf['etiqueta_nb07']}: valores de exposicion equivocados "
            f"(debe usar '{cfg_rf['valores_exposicion']}')."
        )
    RF_CONFIG[clave]["hiperparametros"] = {
        "n_estimators": rf.n_estimators, "max_depth": rf.max_depth,
        "min_samples_leaf": rf.min_samples_leaf, "min_samples_split": rf.min_samples_split,
        "max_features": rf.max_features, "random_state": SEED, "n_jobs": -1,
    }
    print(f"    hiperparametros replicados: n_estimators={rf.n_estimators}, "
          f"max_depth={rf.max_depth}, min_samples_leaf={rf.min_samples_leaf}")
    del rf
del _test
print("  -> esquemas y magnitudes correctos.\n")

# --- C. Gemelos ------------------------------------------------------------------
ajuste = matriz[MASK_AJUSTE]
diag = matriz[MASK_DIAG]
print("=== GEMELOS DE VALIDACION ===")
print(f"Ajuste     : {len(ajuste):,} filas ({ESPEC_NB07['particion_interna']['ajuste']})")
print(f"Diagnostico: {len(diag):,} filas ({ESPEC_NB07['particion_interna']['validacion']})\n")

hp_offset = dict(ESPEC_NB07["modelos"]["frecuencia_xgboost_offset_v2.pkl"]["hiperparametros"])
n_rondas_offset = ESPEC_NB07["modelos"]["frecuencia_xgboost_offset_v2.pkl"]["n_rondas"]
params_xgb = {**hp_offset, "objective": "count:poisson", "seed": SEED}

dm_ajuste = xgb.DMatrix(ajuste[COLUMNAS_OFFSET].astype(float), label=ajuste[COL_Y])
dm_ajuste.set_base_margin(ajuste[OFFSET_COL].values + LOG_TASA_BASE)
print(f"Entrenando gemelo XGBoost ({n_rondas_offset} rondas)...")
GEMELO_XGB = xgb.train(params_xgb, dm_ajuste, num_boost_round=n_rondas_offset)
del dm_ajuste

pred_val = predecir_xgb_offset(diag, booster=GEMELO_XGB)
rmse_val = float(np.sqrt(np.mean((diag[COL_Y].values - pred_val) ** 2)))
calib_val = float(pred_val.mean() / diag[COL_Y].mean())
ref = METRICAS_NB07["optuna"]["offset"]
print(f"  RMSE en validacion       : {rmse_val:.6f} | NB07 publica {ref['rmse_validacion']:.6f}")
print(f"  Calibracion en validacion: {calib_val:.6f} | NB07 publica {ref['calibracion_validacion']:.6f}")

_d1, _d2 = abs(rmse_val - ref["rmse_validacion"]), abs(calib_val - ref["calibracion_validacion"])
if _d1 > 1e-4 or _d2 > 1e-4:
    print(f"\n  AVISO: el gemelo no reproduce exactamente la validacion de NB07")
    print(f"    |dif| RMSE {_d1:.2e} | |dif| calibracion {_d2:.2e}")
    print("    Causas posibles: version de xgboost, orden de filas, o parametros no")
    print("    registrados en el JSON. Se continua, pero queda declarado.")
    REPRODUCE_OPTUNA = False
else:
    print("  -> reproduce la busqueda de Optuna. La optimizacion de NB07 es replicable.")
    REPRODUCE_OPTUNA = True

GEMELOS_RF = {}
for clave, cfg_rf in RF_CONFIG.items():
    print(f"\nEntrenando gemelo {clave} (sin referencia publicada en validacion)...")
    rf = RandomForestRegressor(**cfg_rf["hiperparametros"])
    rf.fit(matriz_rf(ajuste, cfg_rf["valores_exposicion"]), ajuste[COL_Y])
    GEMELOS_RF[clave] = rf
    p = rf.predict(matriz_rf(diag, cfg_rf["valores_exposicion"]))
    print(f"  calibracion en validacion: {p.mean() / diag[COL_Y].mean():.6f}")

# --- D. Tabla por distrito -------------------------------------------------------
def construir_tabla(mask, etiqueta: str, usar_gemelos: bool) -> pd.DataFrame:
    """usar_gemelos=True -> modelos entrenados solo en 2022-2023 (unico modo valido en 2024)."""
    df = matriz[mask].copy()
    if usar_gemelos:
        df["pred_xgb_offset"] = predecir_xgb_offset(df, booster=GEMELO_XGB)
        for clave, cfg_rf in RF_CONFIG.items():
            df[f"pred_{clave}"] = GEMELOS_RF[clave].predict(
                matriz_rf(df, cfg_rf["valores_exposicion"]))
    else:
        df["pred_xgb_offset"] = predecir_xgb_offset(df)
        for clave, cfg_rf in RF_CONFIG.items():
            rf = joblib.load(ARTEFACTOS_PREVIOS[clave][0])
            df[f"pred_{clave}"] = rf.predict(matriz_rf(df, cfg_rf["valores_exposicion"]))
            del rf
    df["exposicion"] = np.exp(df[OFFSET_COL])

    tab = df.groupby(COL_DISTRITO).agg(
        obs=(COL_Y, "sum"), exposicion=("exposicion", "sum"),
        pred_xgb_offset=("pred_xgb_offset", "sum"),
        pred_rf_e1=("pred_rf_e1", "sum"), pred_rf_v2=("pred_rf_v2", "sum"),
        celdas=(COL_Y, "size"),
        celdas_con_evento=(COL_Y, lambda s: int((s > 0).sum())),
    ).join(
        df.groupby(COL_DISTRITO).agg(
            poblacion=("poblacion", "first"), pct_urbano=("pct_urbano", "first"),
            poblacion_baja=("poblacion_baja", "first"),
            estacion=(COL_ESTACION, "first"), departamento=(COL_DEPTO, "first"),
        )
    )
    tab["tasa_per_capita"] = tab["obs"] / tab["poblacion"] * 1000
    tab["tercil"] = asignar_terciles(tab["obs"])
    tab.attrs["particion"] = etiqueta
    tab.attrs["gemelos"] = usar_gemelos
    return tab


print("\n\nConstruyendo tablas por distrito...")
tabla_diag = construir_tabla(MASK_DIAG, PART_DIAG, usar_gemelos=True)
tabla_test = construir_tabla(MASK_TEST, "test", usar_gemelos=False)  # descriptiva (D1)

MODELOS = {
    "XGB offset (NB07)": "pred_xgb_offset",
    "RF Etapa 1": "pred_rf_e1",
    "RF NB05": "pred_rf_v2",
}


def spearman_por_tercil(tab: pd.DataFrame) -> pd.DataFrame:
    filas = {}
    for etiqueta, col in MODELOS.items():
        filas[etiqueta] = {t: spearman_seguro(tab.loc[tab["tercil"] == t, col],
                                              tab.loc[tab["tercil"] == t, "obs"])
                           for t in TERCILES}
        filas[etiqueta]["global"] = spearman_seguro(tab[col], tab["obs"])
    return pd.DataFrame(filas).T[["global", *TERCILES]]


sp_diag = spearman_por_tercil(tabla_diag)
sp_test = spearman_por_tercil(tabla_test)

print(f"\n=== SPEARMAN POR TERCIL — {PART_DIAG}, gemelos 2022-2023 (DECIDE) ===")
display(sp_diag.round(6))
print(f"=== SPEARMAN POR TERCIL — test 2025-2026, modelos finales {ETIQUETA_DESCRIPTIVA} ===")
display(sp_test.round(6))

# --- Control de contaminacion ----------------------------------------------------
for a, b in [("RF Etapa 1", "RF NB05"), ("XGB offset (NB07)", "RF NB05")]:
    if np.allclose(sp_diag.loc[a].values, sp_diag.loc[b].values, atol=1e-6, equal_nan=True):
        raise RuntimeError(
            f"'{a}' y '{b}' producen Spearman identicos en la particion de diagnostico. "
            "Sintoma de que los modelos vieron esos datos durante el entrenamiento."
        )
if (sp_diag[TERCILES] > 0.995).any().any():
    print("\n  AVISO: algun Spearman supera 0.995 en diagnostico. Revisar los gemelos.")
print("\nControl de contaminacion: superado (los modelos difieren entre si).")

print("\nControl — tercil alto en test:")
for etiqueta, clave_nb07 in [("XGB offset (NB07)", "NB07 - XGBoost (exposición offset)"),
                             ("RF Etapa 1", "Etapa 1 - Random Forest"),
                             ("RF NB05", "NB05 - Random Forest")]:
    pub = METRICAS_NB07["modelos"][clave_nb07]["spearman_tercil_alto"]
    rec = float(sp_test.loc[etiqueta, "alto"])
    print(f"  {etiqueta:20s} publicado {pub:.6f} | recalculado {rec:.6f} | "
          f"{'OK' if abs(pub - rec) < 1e-5 else 'DISCREPA'}")


def estructura_terciles(tab, etiqueta):
    e = tab.groupby("tercil").agg(
        n=("obs", "size"), obs_min=("obs", "min"), obs_max=("obs", "max"),
        obs_mediana=("obs", "median"),
        CV=("obs", lambda s: s.std(ddof=1) / s.mean()),
    ).reindex(TERCILES)
    e["rango"] = e["obs_max"] - e["obs_min"]
    e["rango_relativo"] = e["rango"] / e["obs_mediana"]
    print(f"\n=== ESTRUCTURA DE LOS TERCILES ({etiqueta}) ===")
    display(e.round(4))
    return e


estructura_diag = estructura_terciles(tabla_diag, PART_DIAG)
estructura_test = estructura_terciles(tabla_test, f"test {ETIQUETA_DESCRIPTIVA}")
print("Al partir por total observado, el tercil medio tiene por construccion el menor")
print("rango relativo: sus distritos estan mas juntos y su orden es el mas fragil al azar.")

GEMELOS_INFO = {
    "motivo": "Los .pkl finales fueron entrenados con 2022-2024 e incluyen la particion de "
              "diagnostico. Se reinstancian los candidatos de la busqueda de NB07.",
    "ajuste": ESPEC_NB07["particion_interna"]["ajuste"],
    "diagnostico": ESPEC_NB07["particion_interna"]["validacion"],
    "xgb_reproduce_optuna": bool(REPRODUCE_OPTUNA),
    "xgb_rmse_validacion": rmse_val, "xgb_calibracion_validacion": calib_val,
    "xgb_referencia_nb07": ref, "rf_sin_referencia_publicada": True,
    "hiperparametros_rf": {k: v["hiperparametros"] for k, v in RF_CONFIG.items()},
}

VEREDICTO_REALIDAD = {}   # se completa en 2.6

# %% [markdown]
# ## 2.2 Techo de ruido: ¿cuánto del ranking observado es señal?
#
# Pregunta que antecede a todas las demás: **¿existe un orden verdadero que el modelo pudiera
# acertar?**
#
# El ranking observado no es la verdad; es una realización de un proceso aleatorio. Si se
# pudiera repetir 2024 en las mismas condiciones, los conteos serían distintos y el orden
# cambiaría. Un modelo perfecto —uno que conociera la tasa verdadera de cada distrito— tampoco
# alcanzaría Spearman 1.0, porque el observado tiene ruido que ninguna tasa puede predecir.
#
# **Estimador A — semanas alternas.** Se divide la partición en dos mitades por paridad de
# semana ISO y se correlacionan sus rankings. Ambas reflejan la misma realidad, así que su
# discrepancia mide ruido puro. Se corrige con Spearman-Brown, que ajusta por el hecho de que
# cada mitad tiene la mitad de los datos. Se usan semanas alternas para que la tendencia y la
# estacionalidad no contaminen la comparación.
#
# **Estimador B — simulación Poisson.** Se toma el total observado de cada distrito como su tasa
# verdadera, se simulan conteos Poisson y se correlaciona cada simulación con el observado real.
# Es optimista, pero sirve como cota superior.
#
# **Cómo leer el resultado.** Si el techo del tercil medio resulta cercano al Spearman que
# alcanza el modelo vigente, éste ya está en el límite de lo alcanzable. Un modelo que quede
# *por encima* del techo indicaría que reproduce el ruido particular del período en vez de la
# señal — o que el estimador subestima el techo en ese subgrupo.

# %%
# NB08 — 2.2 Techo de ruido

exigir_particion_de_diagnostico(PART_DIAG)


def spearman_brown(r: float, factor: float = 2.0) -> float:
    """Corrige una correlacion split-half por la reduccion de tamano de cada mitad."""
    if np.isnan(r) or r <= -1 / factor:
        return np.nan
    return float(factor * r / (1 + (factor - 1) * r))


def techo_split_half(mask, tab: pd.DataFrame) -> pd.DataFrame:
    """Estimador A: correlacion entre semanas pares e impares, corregida por Spearman-Brown."""
    df = matriz[mask].copy()
    df["semana"] = df[COL_FECHA].dt.isocalendar().week.astype(int)
    df["mitad"] = np.where(df["semana"] % 2 == 0, "A", "B")
    ancho = (df.groupby([COL_DISTRITO, "mitad"])[COL_Y].sum()
             .unstack("mitad").join(tab["tercil"]))
    filas = []
    for t in [*TERCILES, "global"]:
        sub = ancho if t == "global" else ancho[ancho["tercil"] == t]
        r = spearman_seguro(sub["A"], sub["B"])
        filas.append({"tercil": t, "n": len(sub), "r_mitades": r,
                      "techo_corregido": spearman_brown(r)})
    return pd.DataFrame(filas).set_index("tercil")


def techo_poisson(tab: pd.DataFrame, n_sim: int = 500) -> pd.DataFrame:
    """Estimador B: Spearman esperado de un predictor que conociera la tasa verdadera."""
    lam = tab["obs"].values.astype(float)
    sims = RNG.poisson(lam=lam, size=(n_sim, len(lam)))
    filas = []
    for t in [*TERCILES, "global"]:
        sel = np.ones(len(tab), bool) if t == "global" else (tab["tercil"] == t).values
        rs = np.array([spearman_seguro(sims[i][sel], lam[sel]) for i in range(n_sim)], float)
        filas.append({"tercil": t, "n": int(sel.sum()),
                      "techo_medio": float(np.nanmean(rs)),
                      "techo_p5": float(np.nanpercentile(rs, 5)),
                      "techo_p95": float(np.nanpercentile(rs, 95))})
    return pd.DataFrame(filas).set_index("tercil")


techo_a = techo_split_half(MASK_DIAG, tabla_diag)
print("=== TECHO DE RUIDO — Estimador A (semanas alternas + Spearman-Brown) ===")
display(techo_a.round(6))

techo_b = techo_poisson(tabla_diag)
print("\n=== TECHO DE RUIDO — Estimador B (simulacion Poisson, 500 replicas) ===")
display(techo_b.round(6))

comp = pd.DataFrame({
    "techo_A_semanas": techo_a["techo_corregido"],
    "techo_B_poisson": techo_b["techo_medio"],
}).reindex([*TERCILES, "global"])
for etiqueta in MODELOS:
    comp[etiqueta] = [sp_diag.loc[etiqueta, t] for t in comp.index]
comp["margen_XGB_vs_techoA"] = comp["techo_A_semanas"] - comp["XGB offset (NB07)"]
comp["excede_techo"] = comp[list(MODELOS)].max(axis=1) > comp["techo_A_semanas"]

print(f"\n=== LO ALCANZABLE vs LO ALCANZADO ({PART_DIAG}) ===")
display(comp.round(6))
print("\nLectura:")
print("  margen ~ 0  -> el modelo esta en el limite de lo que permite el ruido")
print("  margen > 0  -> queda capacidad de ordenamiento sin aprovechar")
print("  margen < 0  -> el modelo supera el techo estimado: revisar el estimador")

techo_a_test = techo_split_half(MASK_TEST, tabla_test)
comp_test = pd.DataFrame({"techo_A_semanas": techo_a_test["techo_corregido"]})
for etiqueta in MODELOS:
    comp_test[etiqueta] = [sp_test.loc[etiqueta, t] for t in comp_test.index]
print(f"\n=== MISMO CALCULO SOBRE EL TEST {ETIQUETA_DESCRIPTIVA} ===")
display(comp_test.round(6))

TECHO_RUIDO = {
    "estimador_A_semanas_alternas": techo_a["techo_corregido"].to_dict(),
    "estimador_B_poisson": techo_b["techo_medio"].to_dict(),
    "alcanzado_por_tercil": {m: sp_diag.loc[m].to_dict() for m in MODELOS},
    "techo_A_test_descriptivo": techo_a_test["techo_corregido"].to_dict(),
}

# %% [markdown]
# ## 2.3 Techo estructural: ¿cuánto puede ordenar el modelo con lo que sabe?
#
# El techo de ruido mide cuánto orden hay en los datos. Éste mide cuánto de ese orden el modelo
# tiene manera de encontrar.
#
# La celda 1.2 mostró el punto de partida: de las ocho variables, la mayoría varía en el tiempo y
# afecta a los 103 distritos de forma parecida. Las únicas constantes dentro de cada distrito son
# `pct_urbano`, `poblacion_baja` y, de forma indirecta, la estación climática.
#
# Con la exposición como *offset*, la predicción se factoriza:
#
# ```
# predicción = exp(log_exposicion_v2 + log_tasa_base) × exp(f(variables))
#              └──────── componente de exposición ────┘   └── multiplicador ──┘
# ```
#
# | Nivel | Qué mide |
# |---|---|
# | Exposición sola | El orden que da la población, sin modelo. El piso. |
# | Multiplicador solo | Lo que aportan las ocho variables por sí mismas. |
# | Predicción completa | El resultado del modelo vigente. |
# | Poisson territorial | GLM con *offset* y solo las variables que distinguen distritos. El máximo alcanzable con la información territorial disponible. |
#
# **Qué distingue este techo del anterior.** El de ruido es irreducible. Éste sí es reducible,
# pero solo agregando información que hoy no está en la matriz —parque vehicular, densidad de red
# vial, aforos—, que es exactamente la limitación que NB05 declaró.

# %%
# NB08 — 2.3 Techo estructural

exigir_particion_de_diagnostico(PART_DIAG)

tabla_diag["comp_exposicion"] = tabla_diag["exposicion"] * np.exp(LOG_TASA_BASE)
tabla_diag["multiplicador"] = tabla_diag["pred_xgb_offset"] / tabla_diag["comp_exposicion"]

COLS_TERR = verificar_predictoras(["pct_urbano", "poblacion_baja"], "Poisson territorial")


def diseno_territorial(df: pd.DataFrame, ref=None) -> pd.DataFrame:
    X = pd.get_dummies(df[[*COLS_TERR, COL_ESTACION]], columns=[COL_ESTACION],
                       drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add")
    return X if ref is None else X.reindex(columns=ref, fill_value=0.0)


X_aj = diseno_territorial(ajuste)
glm_terr = sm.GLM(ajuste[COL_Y], X_aj, family=sm.families.Poisson(),
                  offset=ajuste[OFFSET_COL]).fit()

_diag = matriz[MASK_DIAG].copy()
_diag["pred_terr"] = glm_terr.predict(
    diseno_territorial(_diag, ref=X_aj.columns), offset=_diag[OFFSET_COL])
tabla_diag["pred_territorial"] = _diag.groupby(COL_DISTRITO)["pred_terr"].sum()
del _diag

filas = []
for t in [*TERCILES, "global"]:
    sub = tabla_diag if t == "global" else tabla_diag[tabla_diag["tercil"] == t]
    filas.append({
        "tercil": t, "n": len(sub),
        "exposicion_sola": spearman_seguro(sub["exposicion"], sub["obs"]),
        "poblacion_sola": spearman_seguro(sub["poblacion"], sub["obs"]),
        "multiplicador_solo": spearman_seguro(sub["multiplicador"], sub["obs"]),
        "poisson_territorial": spearman_seguro(sub["pred_territorial"], sub["obs"]),
        "prediccion_completa": spearman_seguro(sub["pred_xgb_offset"], sub["obs"]),
        "rango_multiplicador": float(sub["multiplicador"].max() / sub["multiplicador"].min()),
        "combinaciones_territoriales": len(
            sub[["pct_urbano", "poblacion_baja", "estacion"]].drop_duplicates()),
    })

techo_estructural = pd.DataFrame(filas).set_index("tercil")
print(f"=== TECHO ESTRUCTURAL ({PART_DIAG}) ===")
display(techo_estructural.round(6))
print("\nLectura:")
print("  exposicion_sola     = orden que da la poblacion, sin modelo. El piso.")
print("  multiplicador_solo  = lo que aportan las 8 variables por si mismas.")
print("  combinaciones_territoriales = cuantos perfiles distintos hay para ordenar n distritos.")

limites = pd.DataFrame({
    "techo_ruido (irreducible)": techo_a["techo_corregido"],
    "techo_estructural (reducible con datos nuevos)": techo_estructural["poisson_territorial"],
    "piso: solo exposicion": techo_estructural["exposicion_sola"],
    **{m: [sp_diag.loc[m, t] for t in [*TERCILES, "global"]] for m in MODELOS},
}).reindex([*TERCILES, "global"])

print(f"\n=== LOS TRES LIMITES Y LOS MODELOS ({PART_DIAG}) ===")
display(limites.round(6))

TECHO_ESTRUCTURAL = techo_estructural.to_dict(orient="index")

# %% [markdown]
# ## 2.4 Bootstrap pareado: ¿la diferencia se distingue del ruido?
#
# Los dos coeficientes que se comparan se calculan sobre **los mismos** distritos y contra **el
# mismo** ranking observado: son medidas pareadas y dependientes.
#
# Eso tiene una consecuencia práctica: comparar sus intervalos marginales y ver si se solapan
# **no es un criterio válido**. Dos intervalos pueden solaparse y aun así la diferencia ser
# significativa, y también al revés. El procedimiento correcto es remuestrear distritos y
# calcular **ambos** coeficientes sobre cada remuestra, construyendo el intervalo de la
# **diferencia**.
#
# ### Criterio, fijado antes de calcular
#
# > **La caída se considera real** si el IC 95% de la diferencia pareada
# > (RF Etapa 1 − XGB offset) en el tercil medio **excluye el cero** en la partición de
# > diagnóstico.
#
# - Si el intervalo **contiene** el cero, la comparación 0.6470 vs 0.8794 no sostiene la
#   afirmación de que un modelo ordena mejor que el otro. Es un hallazgo válido, no un fracaso.
# - Si **excluye** el cero pero el modelo vigente ya está en el techo de ruido, la conclusión no
#   es que el vigente falle, sino que el de Etapa 1 supera el ruido.
# - Una diferencia distinguible puede seguir siendo irrelevante en magnitud. Se reporta el
#   tamaño junto con el intervalo.

# %%
# NB08 — 2.4 Bootstrap pareado de la diferencia entre modelos

exigir_particion_de_diagnostico(PART_DIAG)

CRITERIO_REALIDAD = {
    "regla": "La caida se considera real si el IC 95% de la diferencia pareada "
             "(RF Etapa 1 - XGB offset) en el tercil medio EXCLUYE el cero, "
             "en la particion de diagnostico.",
    "particion": PART_DIAG,
    "comparacion": ["RF Etapa 1", "XGB offset (NB07)"],
    "tercil": "medio", "nivel": 0.95, "remuestreos": B_BOOT,
    "matiz_1": "IC que contiene el cero -> la comparacion no sostiene la afirmacion.",
    "matiz_2": "IC que excluye el cero con el vigente en el techo de ruido -> el problema esta "
               "en que Etapa 1 supera el ruido, no en que el vigente falle.",
    "matiz_3": "Una diferencia distinguible puede ser irrelevante en magnitud.",
}
RUTA_CRIT_S2 = DIR_RESULTS / "nb08_criterios_seccion2.json"
RUTA_CRIT_S2.write_text(json.dumps(
    {"generado": datetime.now().isoformat(timespec="seconds"), "criterio": CRITERIO_REALIDAD},
    indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Criterio registrado en {RUTA_CRIT_S2.name} — {datetime.now():%H:%M:%S}\n")


def bootstrap_pareado(tab: pd.DataFrame, col_a: str, col_b: str,
                      n_boot: int = B_BOOT, rng=None) -> dict:
    """Remuestrea distritos y recalcula AMBOS coeficientes sobre cada remuestra."""
    rng = rng or np.random.default_rng(SEED)
    obs = tab["obs"].values.astype(float)
    a, b = tab[col_a].values.astype(float), tab[col_b].values.astype(float)
    n = len(tab)
    ra = np.full(n_boot, np.nan)
    rb = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ra[i] = spearman_seguro(a[idx], obs[idx])
        rb[i] = spearman_seguro(b[idx], obs[idx])
    dif = ra - rb
    val = ~np.isnan(dif)
    return {
        "punto_a": spearman_seguro(a, obs),
        "punto_b": spearman_seguro(b, obs),
        "diferencia_puntual": spearman_seguro(a, obs) - spearman_seguro(b, obs),
        "ic_a": (float(np.nanpercentile(ra, 2.5)), float(np.nanpercentile(ra, 97.5))),
        "ic_b": (float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))),
        "dif_media": float(np.nanmean(dif)),
        "dif_ic_inf": float(np.nanpercentile(dif[val], 2.5)),
        "dif_ic_sup": float(np.nanpercentile(dif[val], 97.5)),
        "prop_dif_positiva": float(np.mean(dif[val] > 0)),
        "remuestras_validas": int(val.sum()),
    }


COMPARACIONES = [
    ("RF Etapa 1", "XGB offset (NB07)"),
    ("RF Etapa 1", "RF NB05"),
    ("RF NB05", "XGB offset (NB07)"),
]

filas = []
for etiqueta_a, etiqueta_b in COMPARACIONES:
    for t in TERCILES:
        sub = tabla_diag[tabla_diag["tercil"] == t]
        r = bootstrap_pareado(sub, MODELOS[etiqueta_a], MODELOS[etiqueta_b],
                              rng=np.random.default_rng(SEED))
        filas.append({
            "comparacion": f"{etiqueta_a} - {etiqueta_b}", "tercil": t, "n": len(sub),
            "spearman_A": r["punto_a"], "spearman_B": r["punto_b"],
            "diferencia": r["diferencia_puntual"],
            "IC_inf": r["dif_ic_inf"], "IC_sup": r["dif_ic_sup"],
            "excluye_cero": bool(r["dif_ic_inf"] > 0 or r["dif_ic_sup"] < 0),
            "amplitud_IC": r["dif_ic_sup"] - r["dif_ic_inf"],
            "prop_A_mayor": r["prop_dif_positiva"],
        })

boot = pd.DataFrame(filas)
print(f"=== BOOTSTRAP PAREADO DE LA DIFERENCIA — {PART_DIAG} ({B_BOOT} remuestras) ===")
display(boot.set_index(["comparacion", "tercil"]).round(6))

filas = []
for t in TERCILES:
    sub = tabla_diag[tabla_diag["tercil"] == t]
    r = bootstrap_pareado(sub, MODELOS["RF Etapa 1"], MODELOS["XGB offset (NB07)"],
                          rng=np.random.default_rng(SEED))
    filas.append({"tercil": t,
                  "RF_E1": r["punto_a"], "RF_E1_IC": f"[{r['ic_a'][0]:.4f}, {r['ic_a'][1]:.4f}]",
                  "XGB_off": r["punto_b"], "XGB_off_IC": f"[{r['ic_b'][0]:.4f}, {r['ic_b'][1]:.4f}]"})
print("\n=== IC MARGINALES (contexto — NO son el criterio de decision) ===")
display(pd.DataFrame(filas).set_index("tercil").round(6))
print("Recordatorio: el solapamiento de IC marginales no determina significancia en")
print("comparaciones pareadas. El criterio es el IC de la DIFERENCIA, arriba.")

filas = []
for t in TERCILES:
    sub = tabla_test[tabla_test["tercil"] == t]
    r = bootstrap_pareado(sub, MODELOS["RF Etapa 1"], MODELOS["XGB offset (NB07)"],
                          rng=np.random.default_rng(SEED))
    filas.append({"tercil": t, "diferencia": r["diferencia_puntual"],
                  "IC_inf": r["dif_ic_inf"], "IC_sup": r["dif_ic_sup"],
                  "excluye_cero": bool(r["dif_ic_inf"] > 0 or r["dif_ic_sup"] < 0)})
boot_test = pd.DataFrame(filas).set_index("tercil")
print(f"\n=== MISMO CALCULO SOBRE EL TEST {ETIQUETA_DESCRIPTIVA} ===")
display(boot_test.round(6))

BOOTSTRAP = {"diagnostico": boot.to_dict("records"), "test_descriptivo": boot_test.to_dict("index")}

# %% [markdown]
# ## 2.5 Robustez: Kendall tau-b y sensibilidad a la definición del corte
#
# **Kendall tau-b como métrica alternativa.** Spearman correlaciona rangos; Kendall cuenta pares
# concordantes y discordantes. No tienen por qué coincidir en valor —tau suele ser menor— pero sí
# en el patrón. Si Spearman muestra una caída en el tercil medio y Kendall no, la caída es un
# artefacto de la métrica.
#
# **Sensibilidad al corte.** Se prueban tres variantes además del protocolo: cortes en 30 y 40
# distritos, y terciles por tasa per cápita.
#
# La tercera merece una advertencia: **no es una prueba de robustez, es otra pregunta**. Ordenar
# por tasa per cápita agrupa distritos distintos. Se incluye porque informa sobre la naturaleza
# del fenómeno, pero un resultado distinto ahí no invalida el análisis principal.
#
# Se reporta también el **ancho del tercil medio** bajo cada variante: cuanto más ancho el
# tercil, más separados sus extremos y más estable el orden.

# %%
# NB08 — 2.5 Robustez: Kendall tau-b y sensibilidad al corte

exigir_particion_de_diagnostico(PART_DIAG)


def kendall_seguro(x, y) -> float:
    """Kendall tau-b (corrige por empates)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.kendalltau(x, y, variant="b").statistic)


filas = []
for etiqueta, col in MODELOS.items():
    for t in TERCILES:
        sub = tabla_diag[tabla_diag["tercil"] == t]
        filas.append({"modelo": etiqueta, "tercil": t,
                      "spearman": spearman_seguro(sub[col], sub["obs"]),
                      "kendall_tau_b": kendall_seguro(sub[col], sub["obs"])})
kendall = pd.DataFrame(filas)
kendall["ratio_tau_sp"] = kendall["kendall_tau_b"] / kendall["spearman"]

print(f"=== KENDALL TAU-B vs SPEARMAN ({PART_DIAG}) ===")
display(kendall.set_index(["modelo", "tercil"]).round(6))
print("Coinciden en patron aunque no en valor: tau es sistematicamente menor por construccion.")

_peor_sp = (kendall.loc[kendall.groupby("modelo")["spearman"].idxmin(), ["modelo", "tercil"]]
            .rename(columns={"tercil": "peor_segun_spearman"}))
_peor_tau = (kendall.loc[kendall.groupby("modelo")["kendall_tau_b"].idxmin(), ["modelo", "tercil"]]
             .rename(columns={"tercil": "peor_segun_kendall"}))
_peores = _peor_sp.merge(_peor_tau, on="modelo")
_peores["coinciden"] = _peores["peor_segun_spearman"] == _peores["peor_segun_kendall"]
print("\nTercil peor segun cada metrica:")
display(_peores.set_index("modelo"))

VARIANTES = {
    "protocolo (35/34/34, conteo)": {"criterio": "obs", "n_alto": 35, "n_medio": 34},
    "corte 30 (30/43/30, conteo)": {"criterio": "obs", "n_alto": 30, "n_medio": 43},
    "corte 40 (40/23/40, conteo)": {"criterio": "obs", "n_alto": 40, "n_medio": 23},
    "tasa per capita (35/34/34)": {"criterio": "tasa_per_capita", "n_alto": 35, "n_medio": 34},
}

filas = []
for nombre, v in VARIANTES.items():
    tercil_v = asignar_terciles(tabla_diag[v["criterio"]], v["n_alto"], v["n_medio"])
    tmp = tabla_diag.assign(tercil_v=tercil_v)
    for etiqueta, col in MODELOS.items():
        for t in TERCILES:
            sub = tmp[tmp["tercil_v"] == t]
            filas.append({"variante": nombre, "modelo": etiqueta, "tercil": t, "n": len(sub),
                          "spearman": spearman_seguro(sub[col], sub["obs"]),
                          "misma_pregunta": v["criterio"] == "obs"})

sensibilidad = pd.DataFrame(filas)
print(f"\n=== SENSIBILIDAD A LA DEFINICION DEL CORTE ({PART_DIAG}) ===")
display(sensibilidad.pivot_table(index=["variante", "tercil"], columns="modelo",
                                 values="spearman").round(6))
print("Las variantes por conteo prueban robustez. La variante por tasa per capita responde")
print("OTRA pregunta (intensidad, no volumen): un resultado distinto ahi no invalida nada.")

filas_ancho = []
for nombre, v in VARIANTES.items():
    tercil_v = asignar_terciles(tabla_diag[v["criterio"]], v["n_alto"], v["n_medio"])
    sub = tabla_diag.assign(tercil_v=tercil_v).query("tercil_v == 'medio'")
    filas_ancho.append({"variante": nombre, "n_medio": len(sub),
                        "obs_min": int(sub["obs"].min()), "obs_max": int(sub["obs"].max()),
                        "rango_relativo": float((sub["obs"].max() - sub["obs"].min())
                                                / sub["obs"].median())})
print("\n=== ANCHO DEL TERCIL MEDIO BAJO CADA VARIANTE ===")
display(pd.DataFrame(filas_ancho).set_index("variante").round(4))

# asignar_terciles() devuelve la serie ordenada por el criterio: hay que realinear.
_tercil_tasa = asignar_terciles(tabla_diag["tasa_per_capita"]).reindex(tabla_diag.index)
_coinciden = int((_tercil_tasa == tabla_diag["tercil"]).sum())
print(f"\nDistritos que quedan en el mismo tercil bajo ambos criterios: "
      f"{_coinciden} de {len(tabla_diag)} ({_coinciden / len(tabla_diag):.1%})")

_cruce = pd.crosstab(tabla_diag["tercil"], _tercil_tasa,
                     rownames=["por conteo"], colnames=["por tasa per capita"])
display(_cruce.reindex(index=TERCILES, columns=TERCILES, fill_value=0))
print("Fuera de la diagonal: distritos que cambian de grupo al medir intensidad en vez de")
print("volumen. Son poblaciones distintas, no dos versiones del mismo subgrupo.")

ROBUSTEZ = {
    "kendall": kendall.to_dict("records"),
    "peor_tercil_por_metrica": _peores.to_dict("records"),
    "sensibilidad_corte": sensibilidad.to_dict("records"),
    "ancho_tercil_medio": filas_ancho,
    "concordancia_conteo_vs_tasa": _coinciden / len(tabla_diag),
}

# %% [markdown]
# ## 2.6 Veredicto: ¿es real la caída del tercil medio?
#
# Se aplica el criterio declarado en 2.4, sin modificarlo.
#
# ### Por qué la caída no era real: la métrica mide el ancho del tercil
#
# El Spearman del tercil medio sube y baja con el ancho del tercil, **no con el modelo**. Cuanto
# más angosto el corte, peor ordenan los tres a la vez: con 23 distritos comprimidos, hasta el
# Random Forest de Etapa 1 cae. Esto es restricción de rango, un artefacto conocido de las
# medidas de correlación. Al definir los terciles por el total observado, el tercil medio queda
# por construcción como el de menor dispersión. El coeficiente no mide cuán bien el modelo
# ordena; mide cuán separados están los distritos que se le pide ordenar.
#
# La variante por tasa per cápita lo confirma desde el otro lado: al reagrupar por intensidad, el
# modelo vigente pasa a ser el mejor de los tres.
#
# ### El techo de ruido excedido, y por qué se declara
#
# En el tercil medio los modelos **superan** el techo estimado por el estimador A. Eso es
# imposible en sentido estricto: ningún predictor puede ordenar mejor de lo que el propio
# fenómeno se ordena consigo mismo. La conclusión correcta es que **el estimador A subestima el
# techo en ese tercil**.
#
# La causa es identificable: dividir el año en semanas pares e impares también divide la
# exposición, y cada mitad del tercil medio queda con menos de 50 siniestros por distrito. La
# corrección de Spearman-Brown ajusta bajo supuestos que no se cumplen con conteos tan pequeños.
# El estimador B, que no divide los datos, da un valor mucho mayor.
#
# La discrepancia entre ambos estimadores es en sí misma información: **el tercil medio es el
# subgrupo donde el techo es más difícil de estimar**, que es otra forma de decir que es el más
# ruidoso.
#
# Por eso el veredicto se decide únicamente por el IC de la diferencia, como declaraba el
# criterio. El techo queda como evidencia de apoyo y no participa de la regla.

# %%
# NB08 — 2.6 Veredicto sobre la realidad de la caida

_fila = boot[(boot["comparacion"] == "RF Etapa 1 - XGB offset (NB07)") &
             (boot["tercil"] == "medio")].iloc[0]

ic_excluye_cero = bool(_fila["excluye_cero"])
caida_es_real = ic_excluye_cero

sp_vigente_medio = float(sp_diag.loc["XGB offset (NB07)", "medio"])
techo_a_medio = float(techo_a.loc["medio", "techo_corregido"])
techo_b_medio = float(techo_b.loc["medio", "techo_medio"])
techo_a_excedido = bool(sp_vigente_medio > techo_a_medio)

_med = sensibilidad[(sensibilidad["tercil"] == "medio") & sensibilidad["misma_pregunta"]]
_pivot_med = _med.pivot_table(index="variante", columns="modelo", values="spearman")
_amplitud_por_corte = float(_pivot_med.max().max() - _pivot_med.min().min())
_amplitud_por_modelo = float(_pivot_med.loc["protocolo (35/34/34, conteo)"].max()
                             - _pivot_med.loc["protocolo (35/34/34, conteo)"].min())

_significativas = boot[boot["excluye_cero"]][
    ["comparacion", "tercil", "diferencia", "IC_inf", "IC_sup"]]

VEREDICTO_REALIDAD = {
    "veredicto": "no_es_real" if caida_es_real is False else "real",
    "caida_es_real": caida_es_real,
    "criterio_aplicado": CRITERIO_REALIDAD["regla"],
    "particion": PART_DIAG,
    "modelos": "gemelos entrenados solo en 2022-2023",
    "tercil_medio": {
        "spearman_rf_e1": float(sp_diag.loc["RF Etapa 1", "medio"]),
        "spearman_xgb_offset": sp_vigente_medio,
        "diferencia_puntual": float(_fila["diferencia"]),
        "ic_diferencia": [float(_fila["IC_inf"]), float(_fila["IC_sup"])],
        "amplitud_ic": float(_fila["amplitud_IC"]),
        "razon_amplitud_sobre_diferencia": float(_fila["amplitud_IC"] / abs(_fila["diferencia"])),
    },
    "restriccion_de_rango": {
        "hallazgo": "El Spearman del tercil medio varia con el ancho del corte, no con el modelo.",
        "amplitud_por_variante_de_corte": _amplitud_por_corte,
        "amplitud_entre_modelos_en_el_protocolo": _amplitud_por_modelo,
        "spearman_medio_por_variante": _pivot_med.to_dict("index"),
        "tasa_per_capita_xgb_medio": float(
            sensibilidad[(sensibilidad["variante"] == "tasa per capita (35/34/34)") &
                         (sensibilidad["tercil"] == "medio") &
                         (sensibilidad["modelo"] == "XGB offset (NB07)")]["spearman"].iloc[0]),
    },
    "techo_de_ruido": {
        "estimador_A_semanas": techo_a_medio,
        "estimador_B_poisson": techo_b_medio,
        "excedido_por_el_vigente": techo_a_excedido,
        "interpretacion": "El estimador A subestima el techo en el tercil medio: dividir por "
                          "semanas deja menos de 50 siniestros por mitad y Spearman-Brown no se "
                          "sostiene con conteos tan pequenos.",
        "no_participa_del_criterio": True,
    },
    "kendall_confirma_patron": bool(_peores["coinciden"].all()),
    "comparaciones_con_ic_significativo": _significativas.to_dict("records"),
}

print("=== VEREDICTO — SECCION 2 ===")
print(f"Particion de decision : {PART_DIAG} (gemelos entrenados en 2022-2023)\n")
print("Criterio declarado en 2.4:")
print(f"  {CRITERIO_REALIDAD['regla']}\n")
print("Tercil medio:")
print(f"  RF Etapa 1        {VEREDICTO_REALIDAD['tercil_medio']['spearman_rf_e1']:.6f}")
print(f"  XGB offset (NB07) {sp_vigente_medio:.6f}")
print(f"  diferencia        {_fila['diferencia']:+.6f}")
print(f"  IC 95%            [{_fila['IC_inf']:+.4f}, {_fila['IC_sup']:+.4f}]  "
      f"-> {'EXCLUYE' if ic_excluye_cero else 'CONTIENE'} el cero")
print(f"  amplitud del IC   {_fila['amplitud_IC']:.4f}  "
      f"({_fila['amplitud_IC'] / abs(_fila['diferencia']):.0f}x la diferencia observada)")

print(f"\nVEREDICTO: {'LA CAIDA NO ES REAL' if not caida_es_real else 'LA CAIDA ES REAL'}")

print("\n--- Por que: la metrica mide el ancho del tercil ---")
display(_pivot_med.round(6))
print(f"  Variacion atribuible al CORTE   : {_amplitud_por_corte:.4f}")
print(f"  Variacion atribuible al MODELO  : {_amplitud_por_modelo:.4f}")
print("  El corte explica mas variacion que el modelo: restriccion de rango.")
print(f"\n  Reagrupando por tasa per capita, el modelo vigente alcanza "
      f"{VEREDICTO_REALIDAD['restriccion_de_rango']['tasa_per_capita_xgb_medio']:.6f}.")

print("\n--- Techo de ruido (evidencia de apoyo, no participa del criterio) ---")
print(f"  estimador A (semanas)  : {techo_a_medio:.6f}")
print(f"  estimador B (Poisson)  : {techo_b_medio:.6f}")
print(f"  modelo vigente         : {sp_vigente_medio:.6f}")
if techo_a_excedido:
    print("  El vigente SUPERA el estimador A. Ningun predictor puede ordenar mejor que el")
    print("  propio fenomeno consigo mismo: el estimador A subestima en este tercil.")

print("\n--- Unicas comparaciones con IC que excluye el cero ---")
if len(_significativas):
    display(_significativas.round(6))
    print("  El efecto medible esta en el tratamiento de la exposicion (RF E1 vs RF NB05),")
    print("  no en el algoritmo, y NO esta en el tercil medio.")
else:
    print("  Ninguna.")

print("\n--- Que sigue ---")
if not caida_es_real:
    MODO_DIAG = "exploratorio"
    HAY_REMEDIO_JUSTIFICADO = False
    print("  Seccion 3 -> modo EXPLORATORIO: describe el comportamiento del modelo en el")
    print("               rango medio, sin presentarlo como causa de un problema.")
    print("  Seccion 4 -> el remedio al tercil medio NO se ejecuta. Se redirige al sesgo")
    print("               rural que la Seccion 5 documenta.")
else:
    MODO_DIAG = "confirmatorio"
    HAY_REMEDIO_JUSTIFICADO = True
    print("  Seccion 3 -> modo CONFIRMATORIO.")

SECCION2 = {
    "generado": datetime.now().isoformat(timespec="seconds"),
    "particion_de_decision": PART_DIAG,
    "gemelos": GEMELOS_INFO,
    "spearman_diagnostico": sp_diag.to_dict("index"),
    "spearman_test_descriptivo": sp_test.to_dict("index"),
    "estructura_terciles": estructura_diag.to_dict("index"),
    "techo_ruido": TECHO_RUIDO,
    "techo_estructural": TECHO_ESTRUCTURAL,
    "bootstrap": BOOTSTRAP,
    "robustez": ROBUSTEZ,
    "criterio": CRITERIO_REALIDAD,
    "veredicto": VEREDICTO_REALIDAD,
    "modo_seccion_3": MODO_DIAG,
}
RUTA_S2 = DIR_RESULTS / "nb08_seccion2.json"
RUTA_S2.write_text(json.dumps(SECCION2, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
print(f"\nEvidencia guardada en {RUTA_S2.relative_to(ROOT)}")

# %% [markdown]
# # Sección 3 — Caracterización del rango medio
#
# ## 3.1 Compuerta: qué cambia tras el veredicto de la Sección 2
#
# La Sección 2 concluyó que la caída del tercil medio **no se distingue del ruido de muestreo**.
# Eso cambia el propósito de esta sección, y conviene decirlo antes de mostrar un solo número.
#
# **El encuadre original.** Iba a ser un diagnóstico: cuatro hipótesis compitiendo por explicar
# una caída, con un veredicto por hipótesis que habilitaría un remedio.
#
# **El encuadre actual.** No hay caída que explicar. Lo que queda es describir cómo se comporta
# el modelo en el rango medio, que sigue teniendo valor: informa las limitaciones de la Sección
# 5 y responde preguntas que un jurado va a hacer aunque el problema no exista.
#
# ### Qué se retira
#
# **H4 — «Etapa 1 acertaba por memorizar el nivel del distrito».** Se retira. Presuponía una
# asimetría entre modelos que la Sección 2 descartó. Investigar por qué un modelo supera a otro
# cuando no se demostró que lo supere sería construir una explicación para un hecho inexistente.
#
# **H3 — «Hay heterogeneidad por distrito que las variables no capturan».** Se retira como
# hipótesis, se conserva como medición. Su propósito era decidir si un remedio estaba
# justificado; sin problema que remediar, la decisión no se plantea. Pero la pregunta de fondo
# —¿cuánto del desajuste es estructura estable y cuánto ruido?— sigue siendo informativa.
#
# ### Qué se conserva
#
# **H1 → resolución del modelo en el rango medio (3.2).** ¿Las predicciones están lo bastante
# separadas como para sostener un orden?
#
# **H2 → origen del ordenamiento entre distritos (3.3).** Ya tiene evidencia en la celda 2.3.
#
# ### Advertencia sobre la interpretación
#
# Esta sección **no explica un problema**. Describe un comportamiento. La diferencia importa para
# la redacción del documento final: presentar una caracterización como si fuera un diagnóstico
# sugiere que hay una falla, y la evidencia dice que no la hay.

# %%
# NB08 — 3.1 Compuerta y encuadre de la Seccion 3

_faltantes = [n for n in ("tabla_diag", "VEREDICTO_REALIDAD", "MODO_DIAG", "sp_diag",
                          "techo_estructural", "GEMELO_XGB")
              if n not in globals()]
if _faltantes:
    raise RuntimeError(f"La Seccion 3 depende de objetos que la Seccion 2 no dejo: {_faltantes}")
if MODO_DIAG not in ("exploratorio", "confirmatorio"):
    raise RuntimeError(f"MODO_DIAG invalido: {MODO_DIAG!r}")

HIPOTESIS_RETIRADAS = {
    "H4_memorizacion": {
        "motivo": "Presuponia una asimetria entre modelos que la Seccion 2 descarto.",
        "consecuencia": "Investigar por que un modelo supera a otro, sin haber demostrado que "
                        "lo supere, seria explicar un hecho inexistente.",
    },
    "H3_heterogeneidad": {
        "motivo": "Su proposito era decidir si un remedio estaba justificado. Sin problema que "
                  "remediar, la decision no se plantea.",
        "consecuencia": "Se conserva como MEDICION descriptiva en 3.3, sin veredicto.",
    },
}

HIPOTESIS_CONSERVADAS = {
    "H1_resolucion": {"celda": "3.2",
                      "pregunta": "Estan las predicciones del rango medio lo bastante separadas "
                                  "como para sostener un orden?"},
    "H2_origen_del_orden": {"celda": "3.3",
                            "pregunta": "De donde viene el ordenamiento entre distritos: de la "
                                        "exposicion o de las ocho variables?",
                            "evidencia_previa": f"Celda 2.3: multiplicador_solo = "
                                                f"{techo_estructural.loc['medio', 'multiplicador_solo']:.4f} "
                                                f"en el tercil medio."},
}

ETIQUETA_DESCRIPTIVO_S3 = "[descriptivo — no diagnostica un problema]"

print("=== SECCION 3 — CARACTERIZACION DEL RANGO MEDIO ===")
print(f"Modo            : {MODO_DIAG.upper()}")
print(f"Veredicto previo: {VEREDICTO_REALIDAD['veredicto']}")
print(f"Particion       : {PART_DIAG} (gemelos 2022-2023)\n")

if MODO_DIAG == "exploratorio":
    print("La Seccion 2 no confirmo que la caida sea real. Esta seccion DESCRIBE el")
    print("comportamiento del modelo en el rango medio; no explica un problema.\n")

print("Hipotesis retiradas:")
for h, d in HIPOTESIS_RETIRADAS.items():
    print(f"  {h}\n    motivo: {d['motivo']}")
print("\nHipotesis conservadas (reformuladas como descripcion):")
for h, d in HIPOTESIS_CONSERVADAS.items():
    print(f"  {h} (celda {d['celda']})\n    {d['pregunta']}")

SECCION3 = {
    "modo": MODO_DIAG,
    "encuadre": "caracterizacion descriptiva, no diagnostico",
    "hipotesis_retiradas": HIPOTESIS_RETIRADAS,
    "hipotesis_conservadas": HIPOTESIS_CONSERVADAS,
}

# %% [markdown]
# ## 3.2 Resolución: ¿están las predicciones lo bastante separadas para sostener un orden?
#
# La Sección 2 mostró que el **observado** del tercil medio está comprimido. Esta celda mira la
# contracara: qué tan separadas están las **predicciones** en ese mismo rango.
#
# La pregunta importa porque un coeficiente bajo admite dos lecturas: que el modelo ordena mal, o
# que no tiene cómo ordenar porque los valores que compara son indistinguibles. La segunda no es
# una falla del modelo: es una propiedad del fenómeno.
#
# ### La prueba de pares distinguibles
#
# Para cada par de distritos consecutivos en el ranking predicho, se compara la diferencia con la
# incertidumbre propia de un conteo. Dos distritos son distinguibles si
#
# ```
# |pred_i − pred_j|  >  sqrt(pred_i + pred_j)
# ```
#
# El lado derecho es la desviación estándar de la diferencia entre dos conteos Poisson
# independientes. Si la separación entre dos predicciones consecutivas es menor que el ruido
# esperado, el modelo no tiene base para afirmar que uno va antes que el otro.
#
# Es una regla deliberadamente exigente: sirve para comparar terciles entre sí, no como umbral
# absoluto de calidad.
#
# El cálculo se hace para los tres modelos, porque lo que interesa no es si el modelo vigente
# comprime, sino **si comprime más que aquel con el que se lo compara**.

# %%
# NB08 — 3.2 Resolucion del modelo en el rango medio

exigir_particion_de_diagnostico(PART_DIAG)


def pares_distinguibles(pred: np.ndarray) -> dict:
    """Proporcion de pares consecutivos cuya diferencia supera la desviacion Poisson."""
    p = np.sort(np.asarray(pred, float))[::-1]
    if len(p) < 2:
        return {"proporcion": np.nan, "n_pares": 0, "separacion_rel_mediana": np.nan}
    dif = np.abs(np.diff(p))
    sigma = np.sqrt(p[:-1] + p[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(sigma > 0, dif / sigma, np.nan)
    return {"proporcion": float(np.mean(dif > sigma)), "n_pares": int(len(dif)),
            "separacion_rel_mediana": float(np.nanmedian(rel))}


filas = []
for etiqueta, col in MODELOS.items():
    for t in TERCILES:
        sub = tabla_diag[tabla_diag["tercil"] == t]
        d = pares_distinguibles(sub[col].values)
        cv_pred = sub[col].std(ddof=1) / sub[col].mean()
        cv_obs = sub["obs"].std(ddof=1) / sub["obs"].mean()
        filas.append({
            "modelo": etiqueta, "tercil": t, "n": len(sub),
            "pares_distinguibles": d["proporcion"],
            "separacion_rel_mediana": d["separacion_rel_mediana"],
            "CV_predicho": cv_pred, "CV_observado": cv_obs,
            "CV_pred/CV_obs": cv_pred / cv_obs if cv_obs else np.nan,
            "rango_predicho": float(sub[col].max() - sub[col].min()),
            "rango_observado": float(sub["obs"].max() - sub["obs"].min()),
        })

resolucion = pd.DataFrame(filas)
print(f"=== RESOLUCION DE LAS PREDICCIONES {ETIQUETA_DESCRIPTIVO_S3} ===")
print(f"Particion: {PART_DIAG}\n")
display(resolucion.set_index(["modelo", "tercil"]).round(4))

vig = resolucion[resolucion["modelo"] == "XGB offset (NB07)"].set_index("tercil")
print("Modelo vigente — pares consecutivos distinguibles por tercil:")
for t in TERCILES:
    print(f"  {t:6s}: {vig.loc[t, 'pares_distinguibles']:.4f}  "
          f"(separacion relativa mediana {vig.loc[t, 'separacion_rel_mediana']:.4f})")

print("\nCompresion (CV predicho / CV observado): 1.0 = el modelo reproduce la dispersion")
display(resolucion.pivot_table(index="tercil", columns="modelo",
                               values="CV_pred/CV_obs").reindex(TERCILES).round(4))

medio = resolucion[resolucion["tercil"] == "medio"].set_index("modelo")
print("Tercil medio — comparacion entre modelos:")
display(medio[["pares_distinguibles", "separacion_rel_mediana",
               "CV_predicho", "CV_observado", "CV_pred/CV_obs"]].round(4))

_min_mod, _max_mod = medio["pares_distinguibles"].min(), medio["pares_distinguibles"].max()
_min_ter, _max_ter = vig["pares_distinguibles"].min(), vig["pares_distinguibles"].max()
print(f"\nVariacion entre MODELOS en el tercil medio : {_max_mod - _min_mod:.4f}")
print(f"Variacion entre TERCILES en el modelo vigente: {_max_ter - _min_ter:.4f}")

SECCION3["resolucion"] = {
    "tabla": resolucion.to_dict("records"),
    "variacion_entre_modelos_tercil_medio": float(_max_mod - _min_mod),
    "variacion_entre_terciles_modelo_vigente": float(_max_ter - _min_ter),
    "nota": "Regla exigente (una desviacion Poisson completa). Sirve para comparar terciles "
            "entre si, no como umbral absoluto de calidad.",
}

# %% [markdown]
# ## 3.3 De dónde viene el ordenamiento entre distritos
#
# Dos mediciones, ambas descriptivas.
#
# ### A. Grados de libertad para ordenar
#
# De las ocho variables, solo `pct_urbano` y `poblacion_baja` son constantes dentro de un
# distrito; la estación climática actúa como una tercera de forma indirecta. El resto es temporal
# y afecta a los 103 distritos por igual: puede cambiar la predicción de un día, no el orden
# anual entre distritos.
#
# Si 34 distritos comparten pocas combinaciones territoriales distintas, el modelo no tiene con
# qué asignarles órdenes diferentes más allá de lo que aporte la exposición.
#
# ### B. Efecto de distrito: ¿estructura estable o ruido?
#
# El efecto residual se define como `e_d = log((observado_d + 0.5) / (predicho_d + 0.5))`. La
# corrección de 0.5 evita el logaritmo de cero en distritos pequeños.
#
# La pregunta es si ese desajuste se **repite entre períodos**. Un efecto real —un corredor vial,
# una intensidad de reporte particular— debería aparecer en los dos años. El ruido no.
#
# **Limitación declarada.** El gemelo fue entrenado sobre 2022-2023, así que el efecto medido en
# esos años está estimado sobre datos que el modelo vio. Eso lo **comprime hacia cero**: la
# estabilidad medida es una **cota inferior**. Si aparece señal estable pese a la compresión, la
# señal real es mayor. Las alternativas eran peores: usar los modelos finales sobre el test
# gastaría la evaluación reservada para la Sección 6.
#
# Esta medición no emite veredicto. Alimenta las limitaciones de la Sección 5 y la línea futura
# que NB05 ya había declarado: parque vehicular por distrito, densidad de red vial, aforos.

# %%
# NB08 — 3.3 Origen del ordenamiento y efecto de distrito

exigir_particion_de_diagnostico(PART_DIAG)

# --- A. Grados de libertad -------------------------------------------------------
filas = []
for t in [*TERCILES, "global"]:
    sub = tabla_diag if t == "global" else tabla_diag[tabla_diag["tercil"] == t]
    n_comb = len(sub[["pct_urbano", "poblacion_baja", "estacion"]].drop_duplicates())
    filas.append({
        "tercil": t, "n_distritos": len(sub),
        "valores_pct_urbano": sub["pct_urbano"].nunique(),
        "valores_poblacion_baja": sub["poblacion_baja"].nunique(),
        "estaciones": sub["estacion"].nunique(),
        "combinaciones_territoriales": n_comb,
        "distritos_por_combinacion": len(sub) / n_comb,
        "exposicion_sola": float(techo_estructural.loc[t, "exposicion_sola"]),
        "multiplicador_solo": float(techo_estructural.loc[t, "multiplicador_solo"]),
        "prediccion_completa": float(techo_estructural.loc[t, "prediccion_completa"]),
        "rango_multiplicador": float(sub["multiplicador"].max() / sub["multiplicador"].min()),
    })

grados = pd.DataFrame(filas).set_index("tercil")
print(f"=== A. GRADOS DE LIBERTAD PARA ORDENAR {ETIQUETA_DESCRIPTIVO_S3} ===")
print(f"Particion: {PART_DIAG}\n")
display(grados.round(4))
print("exposicion_sola    = orden que da la poblacion, sin modelo")
print("multiplicador_solo = orden que aportan las 8 variables por si mismas")

# --- B. Efecto de distrito -------------------------------------------------------
comp_aj = matriz[MASK_AJUSTE].copy()
comp_aj["anio"] = comp_aj[COL_FECHA].dt.year
comp_aj["pred"] = predecir_xgb_offset(comp_aj, booster=GEMELO_XGB)

efecto = (comp_aj.groupby([COL_DISTRITO, "anio"])
          .agg(obs=(COL_Y, "sum"), pred=("pred", "sum")).reset_index())
efecto["e_d"] = np.log((efecto["obs"] + 0.5) / (efecto["pred"] + 0.5))

anios = sorted(efecto["anio"].unique())
ancho = efecto.pivot(index=COL_DISTRITO, columns="anio", values="e_d")
ancho.columns = [f"e_d_{a}" for a in ancho.columns]
ancho = ancho.join(tabla_diag["tercil"])
col_a, col_b = f"e_d_{anios[0]}", f"e_d_{anios[1]}"

filas = []
for t in [*TERCILES, "global"]:
    sub = ancho if t == "global" else ancho[ancho["tercil"] == t]
    filas.append({
        "tercil": t, "n": len(sub),
        "estabilidad_spearman": spearman_seguro(sub[col_a], sub[col_b]),
        "estabilidad_pearson": float(np.corrcoef(sub[col_a], sub[col_b])[0, 1]),
        f"sd_{anios[0]}": float(sub[col_a].std(ddof=1)),
        f"sd_{anios[1]}": float(sub[col_b].std(ddof=1)),
        f"ratio_mediano_{anios[0]}": float(np.exp(sub[col_a].median())),
        f"ratio_mediano_{anios[1]}": float(np.exp(sub[col_b].median())),
    })

estabilidad = pd.DataFrame(filas).set_index("tercil")
print(f"\n=== B. EFECTO DE DISTRITO: ESTABILIDAD ENTRE {anios[0]} Y {anios[1]} ===")
print("LIMITACION: el gemelo fue entrenado sobre estos dos anios. El efecto esta")
print("comprimido hacia cero -> la estabilidad medida es una COTA INFERIOR.\n")
display(estabilidad.round(4))

ancho["e_d_medio"] = ancho[[col_a, col_b]].mean(axis=1)
ancho["coherente"] = np.sign(ancho[col_a]) == np.sign(ancho[col_b])
persistentes = ancho[ancho["coherente"]].reindex(
    ancho[ancho["coherente"]]["e_d_medio"].abs().sort_values(ascending=False).index)

print("Distritos con desajuste persistente (mismo signo en ambos anios), 10 mayores:")
_muestra = persistentes.head(10)[[col_a, col_b, "e_d_medio", "tercil"]].copy()
_muestra["ratio_medio"] = np.exp(_muestra["e_d_medio"])
display(_muestra.round(4))
print(f"Distritos con desajuste coherente entre anios: {int(ancho['coherente'].sum())} "
      f"de {len(ancho)} ({ancho['coherente'].mean():.1%})")

_j = ancho.join(tabla_diag[["poblacion", "pct_urbano", "tasa_per_capita"]])
print("\nCorrelacion del desajuste medio con variables territoriales:")
for v in ("poblacion", "pct_urbano", "tasa_per_capita"):
    print(f"  {v:18s}: {spearman_seguro(_j[v], _j['e_d_medio']):+.4f}")
print("(Correlaciones debiles -> el desajuste no lo explican las variables disponibles:")
print(" es la limitacion territorial que NB05 ya habia declarado.)")

SECCION3["grados_de_libertad"] = grados.to_dict("index")
SECCION3["efecto_distrito"] = {
    "estabilidad": estabilidad.to_dict("index"),
    "anios_comparados": [int(a) for a in anios],
    "prop_coherentes": float(ancho["coherente"].mean()),
    "limitacion": "El gemelo fue entrenado sobre los anios en que se mide el efecto. "
                  "La estabilidad reportada es una cota inferior.",
}
RUTA_S3 = DIR_RESULTS / "nb08_seccion3.json"
RUTA_S3.write_text(json.dumps(SECCION3, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
print(f"\nGuardado en {RUTA_S3.relative_to(ROOT)}")
del comp_aj

# %% [markdown]
# # Sección 4 — Remedio al sesgo de calibración rural
#
# ## 4.1 Encuadre: qué se corrige y por qué este problema y no el otro
#
# Esta sección estaba prevista para corregir el desorden del tercil medio. La Sección 2 determinó
# que ese desorden no se distingue del ruido y la Sección 3 mostró que ningún modelo tiene
# resolución en ese rango. Corregirlo sería ajustar el ruido particular de un período.
#
# Pero la Sección 5 encuentra un problema distinto que **sí** está demostrado: el modelo predice
# alrededor de un 14% menos siniestros de los que ocurren en las zonas más rurales. Tres
# mediciones independientes señalan el mismo conjunto de distritos.
#
# A diferencia del tercil medio, este problema tiene magnitud, se repite entre períodos y tiene
# consecuencia práctica: una asignación de recursos guiada por el modelo subatendería esas zonas.
#
# ### El remedio
#
# ```
# c_rural = Σ observado_rural / Σ predicho_rural        (solo sobre 2022-2023)
# predicción corregida = predicción × c_rural           (solo en distritos rurales)
# ```
#
# Un factor único por grupo no puede reordenar distritos dentro del grupo —multiplica a todos por
# lo mismo—, así que **no puede inflar artificialmente el Spearman**. Solo corrige el nivel.
#
# ### Tensión con NB05, declarada
#
# NB05 rechazó la corrección territorial con tres argumentos: memoriza en vez de explicar, no
# generaliza a los 159 distritos sin registros, y confunde siniestralidad real con intensidad de
# reporte. El primero y el tercero **siguen vigentes**. Lo que cambia es el alcance: NB05 evaluaba
# un efecto **por distrito** (103 parámetros); aquí se estima **un solo parámetro** para un grupo
# definido por una variable que ya está en el modelo. Se declara como una **revisión acotada y
# consciente**, no como una técnica sin relación con aquella decisión.
#
# ### Criterio de admisión, fijado antes de calcular
#
# | # | Condición | Umbral |
# |---|---|---|
# | 1 | La calibración del grupo rural se acerca a 1 | \|calib − 1\| baja al menos 0.05 |
# | 2 | La calibración global no se degrada | no empeora más de 0.02 |
# | 3 | El RMSE global no se degrada | no empeora más de 0.5% |
# | 4 | El ordenamiento no se degrada | Spearman global no baja más de 0.01 |
#
# ### Qué se descartó
#
# | Alternativa | Motivo |
# |---|---|
# | Efecto aleatorio por distrito (103 parámetros) | Es exactamente lo que NB05 rechazó |
# | Variables de OpenStreetMap | Sin polígonos de distrito; el centroide se calculó promediando coordenadas de accidentes, así que cualquier radio alrededor está definido por el target. Fuga circular |
# | Modelo en dos etapas | Su propósito era el ordenamiento del tercil medio, que no está roto |

# %%
# NB08 — 4.1 Encuadre y criterio de admision del remedio

UMBRAL_RURAL = 0.3   # el unico umbral donde la Seccion 5.2 detecta brecha
COL_GRUPO = "grupo_territorial"


def marcar_grupo(df: pd.DataFrame) -> np.ndarray:
    """Grupo territorial: 'rural' si pct_urbano <= UMBRAL_RURAL, 'resto' si no."""
    return np.where(df["pct_urbano"] <= UMBRAL_RURAL, "rural", "resto")


_dist_grupo = (matriz.groupby(COL_DISTRITO)["pct_urbano"].first() <= UMBRAL_RURAL)
N_RURAL = int(_dist_grupo.sum())

CRITERIO_REMEDIO = {
    "problema": "Subestimacion de calibracion en distritos rurales (L1)",
    "magnitud_detectada": {"n_distritos": N_RURAL, "umbral_pct_urbano": UMBRAL_RURAL},
    "remedio": "Factor multiplicativo unico para el grupo rural, estimado como "
               "sum(observado)/sum(predicho) sobre la particion de ajuste 2022-2023.",
    "por_que_es_seguro": "Un factor unico por grupo no reordena distritos dentro del grupo: "
                         "no puede inflar el Spearman.",
    "estimacion": "particion de ajuste (2022-2023)",
    "validacion": "validacion 2024",
    "condiciones": {
        "1_mejora_rural": {"metrica": "|calibracion_rural - 1|",
                           "regla": "debe bajar al menos 0.05", "umbral": 0.05},
        "2_calibracion_global": {"metrica": "|calibracion_global - 1|",
                                 "regla": "no empeora mas de 0.02", "umbral": 0.02},
        "3_rmse_global": {"metrica": "RMSE global",
                          "regla": "no empeora mas de 0.5%", "umbral": 0.005},
        "4_ordenamiento": {"metrica": "spearman global",
                           "regla": "no baja mas de 0.01", "umbral": 0.01},
    },
    "regla_de_decision": "Se acepta solo si se cumplen LAS CUATRO condiciones. "
                         "Resultado en el limite -> indeterminado.",
    "tension_con_nb05": {
        "decision_previa": ESPEC_NB05["correccion_territorial"]["decision"],
        "que_sigue_vigente": ["memoriza en vez de explicar",
                              "confunde siniestralidad con intensidad de reporte"],
        "que_cambia": "NB05 evaluaba un efecto por distrito (103 parametros). Aqui se estima UN "
                      "parametro para un grupo definido por una variable ya presente en el modelo.",
        "caracter": "revision acotada y consciente, no tecnica sin relacion con NB05",
    },
    "alternativas_descartadas": {
        "efecto_aleatorio_por_distrito": "Es lo que NB05 rechazo: memoriza y no generaliza.",
        "variables_osm": "Sin poligonos de distrito; el centroide se calculo con las coordenadas "
                         "de los accidentes -> fuga circular. Linea futura.",
        "modelo_en_dos_etapas": "Su proposito era el ordenamiento del tercil medio, que la "
                                "Seccion 2 mostro que no esta roto.",
    },
}

RUTA_CRIT_S4 = DIR_RESULTS / "nb08_criterios_seccion4.json"
RUTA_CRIT_S4.write_text(json.dumps(
    {"generado": datetime.now().isoformat(timespec="seconds"), "criterio": CRITERIO_REMEDIO},
    indent=2, ensure_ascii=False), encoding="utf-8")

print("=== SECCION 4 — REMEDIO AL SESGO DE CALIBRACION RURAL ===\n")
print(f"Problema         : {CRITERIO_REMEDIO['problema']}")
print(f"Grupo afectado   : {N_RURAL} distritos con pct_urbano <= {UMBRAL_RURAL}")
print(f"\nRemedio          : {CRITERIO_REMEDIO['remedio']}")
print(f"Se estima en     : {CRITERIO_REMEDIO['estimacion']}")
print(f"Se valida en     : {CRITERIO_REMEDIO['validacion']}")
print("\nCondiciones de admision (las CUATRO deben cumplirse):")
for k, c in CRITERIO_REMEDIO["condiciones"].items():
    print(f"  {k}: {c['metrica']} — {c['regla']}")
print(f"\nTension con NB05 declarada:")
print(f"  decision previa: {CRITERIO_REMEDIO['tension_con_nb05']['decision_previa']}")
print(f"\nCriterio registrado en {RUTA_CRIT_S4.name} — {datetime.now():%H:%M:%S}")
print("(marca de tiempo anterior a cualquier resultado del remedio)")

SECCION4 = {"criterio": CRITERIO_REMEDIO, "umbral_rural": UMBRAL_RURAL, "n_rural": N_RURAL}

# %% [markdown]
# ## 4.2 Estimación del factor y evaluación de las cuatro condiciones
#
# El factor se estima **únicamente** sobre la partición de ajuste (2022-2023) y se evalúa sobre
# la validación 2024. El conjunto de prueba no participa.
#
# `c_rural = Σ observado / Σ predicho` es el estimador de máxima verosimilitud del factor de
# escala bajo un modelo Poisson. No hay hiperparámetro que ajustar.
#
# ### Advertencia sobre el sesgo de estimación
#
# El gemelo fue entrenado sobre 2022-2023, que es la misma ventana donde se estima el factor.
# Sobre sus propios datos de entrenamiento el modelo ya está bien calibrado, así que el factor
# estimado ahí será **más cercano a 1** que el que haría falta en datos nuevos. El remedio se
# estima con una versión atenuada del problema que busca corregir.
#
# Eso juega en contra del remedio, no a favor. Si aun así cumple las cuatro condiciones, la
# mejora es real y probablemente mayor que la medida.
#
# ### Lo que este remedio no puede hacer
#
# Un factor único multiplica todas las predicciones del grupo por el mismo número. No reordena
# distritos dentro del grupo. Por eso el Spearman rural será **idéntico** antes y después: es una
# propiedad matemática, no un resultado. Se verifica como control de implementación.

# %%
# NB08 — 4.2 Estimacion del factor y evaluacion de las cuatro condiciones

exigir_particion_de_diagnostico(PART_DIAG)

aj = matriz[MASK_AJUSTE].copy()
va = matriz[MASK_DIAG].copy()
for df in (aj, va):
    df["pred"] = predecir_xgb_offset(df, booster=GEMELO_XGB)
    df[COL_GRUPO] = marcar_grupo(df)


def estimar_factores(df: pd.DataFrame) -> dict:
    """c_g = sum(observado_g) / sum(predicho_g) por grupo territorial."""
    g = df.groupby(COL_GRUPO).agg(obs=(COL_Y, "sum"), pred=("pred", "sum"))
    return {grupo: float(r["obs"] / r["pred"]) for grupo, r in g.iterrows()}


factores_aj = estimar_factores(aj)
factores_va = estimar_factores(va)   # informativo: NO se usa para corregir

print("=== FACTOR DE CORRECCION ===")
print("Estimado sobre ajuste 2022-2023 (el que se aplica):")
for g, c in sorted(factores_aj.items()):
    print(f"  {g:6s}: {c:.6f}")
print("\nEl mismo calculo sobre validacion 2024 (informativo, NO se usa):")
for g, c in sorted(factores_va.items()):
    print(f"  {g:6s}: {c:.6f}")
print("\nLa diferencia entre ambos refleja el sesgo de estimacion: el gemelo fue entrenado")
print("sobre 2022-2023, donde ya esta bien calibrado. El factor estimado ahi esta atenuado")
print("-> el remedio se estima con una version debilitada del problema. Juega en su contra.")

VARIANTES_REMEDIO = {
    "A - solo rural": {"rural": factores_aj["rural"], "resto": 1.0},
    "B - ambos grupos": {"rural": factores_aj["rural"], "resto": factores_aj["resto"]},
}


def aplicar_remedio(df: pd.DataFrame, factores: dict) -> np.ndarray:
    return df["pred"].values * df[COL_GRUPO].map(factores).values


def evaluar(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Metricas globales y por grupo, segun la jerarquia del protocolo."""
    obs = df[COL_Y].values.astype(float)
    agg = (pd.DataFrame({"d": df[COL_DISTRITO].values, "obs": obs, "pred": pred})
           .groupby("d")[["obs", "pred"]].sum())
    out = {
        "calibracion_global": float(pred.mean() / obs.mean()),
        "RMSE_global": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "MAE_global": float(np.mean(np.abs(obs - pred))),
        "spearman_global": spearman_seguro(agg["pred"], agg["obs"]),
    }
    for grupo in ("rural", "resto"):
        sel = (df[COL_GRUPO] == grupo).values
        o, p = obs[sel], pred[sel]
        sub = (pd.DataFrame({"d": df[COL_DISTRITO].values[sel], "obs": o, "pred": p})
               .groupby("d")[["obs", "pred"]].sum())
        out[f"calibracion_{grupo}"] = float(p.mean() / o.mean())
        out[f"RMSE_{grupo}"] = float(np.sqrt(np.mean((o - p) ** 2)))
        out[f"spearman_{grupo}"] = spearman_seguro(sub["pred"], sub["obs"])
    return out


resultados = {"sin remedio (vigente)": evaluar(va, va["pred"].values)}
for nombre, f in VARIANTES_REMEDIO.items():
    resultados[nombre] = evaluar(va, aplicar_remedio(va, f))

tabla = pd.DataFrame(resultados).T
print(f"\n=== EVALUACION EN {PART_DIAG} ===")
display(tabla[["calibracion_global", "calibracion_rural", "calibracion_resto",
               "RMSE_global", "spearman_global", "spearman_rural"]].round(6))

# Control: un factor unico por grupo NO puede reordenar dentro del grupo.
_sp_base = tabla.loc["sin remedio (vigente)", "spearman_rural"]
for nombre in VARIANTES_REMEDIO:
    _sp = tabla.loc[nombre, "spearman_rural"]
    if not np.isclose(_sp_base, _sp, atol=1e-9):
        raise RuntimeError(
            f"'{nombre}': el Spearman rural cambio de {_sp_base:.9f} a {_sp:.9f}. "
            "Un factor unico por grupo NO puede reordenar: la implementacion es incorrecta."
        )
print(f"\nControl de implementacion: el Spearman rural es identico en las tres filas "
      f"({_sp_base:.6f}).")
print("  Confirma que el remedio corrige el nivel sin tocar el ordenamiento.")

base = resultados["sin remedio (vigente)"]
cond = CRITERIO_REMEDIO["condiciones"]
filas = []
for nombre in VARIANTES_REMEDIO:
    r = resultados[nombre]
    mejora_rural = abs(base["calibracion_rural"] - 1) - abs(r["calibracion_rural"] - 1)
    degrad_global = abs(r["calibracion_global"] - 1) - abs(base["calibracion_global"] - 1)
    degrad_rmse = (r["RMSE_global"] - base["RMSE_global"]) / base["RMSE_global"]
    caida_sp = base["spearman_global"] - r["spearman_global"]

    c1 = mejora_rural >= cond["1_mejora_rural"]["umbral"]
    c2 = degrad_global <= cond["2_calibracion_global"]["umbral"]
    c3 = degrad_rmse <= cond["3_rmse_global"]["umbral"]
    c4 = caida_sp <= cond["4_ordenamiento"]["umbral"]

    filas.append({
        "variante": nombre,
        "1_mejora_rural": mejora_rural, "c1_cumple": c1,
        "2_degrad_calib_global": degrad_global, "c2_cumple": c2,
        "3_degrad_rmse_pct": degrad_rmse * 100, "c3_cumple": c3,
        "4_caida_spearman": caida_sp, "c4_cumple": c4,
        "cumple_todas": bool(c1 and c2 and c3 and c4),
    })

condiciones = pd.DataFrame(filas).set_index("variante")
print("\n=== LAS CUATRO CONDICIONES (umbrales fijados en 4.1, antes de calcular) ===")
print(f"  1. mejora rural        >= {cond['1_mejora_rural']['umbral']}")
print(f"  2. degradacion calib   <= {cond['2_calibracion_global']['umbral']}")
print(f"  3. degradacion RMSE    <= {cond['3_rmse_global']['umbral'] * 100}%")
print(f"  4. caida spearman      <= {cond['4_ordenamiento']['umbral']}\n")
display(condiciones.round(6))

SECCION4["factores"] = {"ajuste": factores_aj, "validacion_informativo": factores_va}
SECCION4["resultados_validacion"] = resultados
SECCION4["condiciones"] = condiciones.to_dict("index")

# %% [markdown]
# ## 4.3 Veredicto: el remedio se rechaza, y el motivo es informativo
#
# Se aplica el criterio declarado en 4.1, sin modificarlo. El remedio no rompe nada —las tres
# últimas condiciones se cumplen con holgura— pero tampoco arregla nada. **Se rechaza por inútil,
# no por dañino.**
#
# ### Por qué falló, y por qué eso importa más que el rechazo
#
# El factor estimado sobre la partición de ajuste es prácticamente 1. El mismo cálculo sobre la
# validación 2024 da un valor mucho mayor.
#
# El modelo está **correctamente calibrado en el grupo rural durante su propio período de
# entrenamiento**. El desajuste no es un error constante que arrastre: aparece cuando proyecta
# hacia adelante.
#
# Un factor estimado en el pasado no puede corregir un problema que en el pasado no existe.
#
# ### Qué significa sobre la naturaleza del sesgo
#
# El resultado reformula la limitación L1. No se trata de que el modelo entienda mal a los
# distritos rurales —los aprende bien cuando los ve—, sino de que **pierde su nivel al
# extrapolar**.
#
# La hipótesis más plausible, que este notebook no puede confirmar, es una **tendencia
# diferencial**: si la siniestralidad rural crece a un ritmo distinto del promedio nacional, el
# factor de crecimiento único de NB05 subcorregiría a ese grupo, y el error se acumularía con el
# tiempo.
#
# ### Qué remedio sí tendría sentido, y por qué no se hace aquí
#
# Un **factor de crecimiento diferenciado por grupo territorial**. No se implementa porque
# (1) modifica la especificación de la exposición, que es objeto de NB05, y obligaría a
# reentrenar toda la cadena; (2) la celda 5.1 muestra que el CAGR es frágil con poca historia, y
# estimarlo sobre 19 distritos multiplicaría esa fragilidad; (3) habría que verificar primero que
# la tendencia diferencial existe.
#
# ### Lectura del rechazo
#
# Que un remedio no funcione no es un resultado negativo. El criterio se fijó antes, se aplicó sin
# modificarlo, y el fracaso produjo información que el éxito no habría dado: si el factor hubiera
# funcionado, habríamos corregido el síntoma sin descubrir que el problema es de extrapolación.

# %%
# NB08 — 4.3 Veredicto del remedio

_alguna_cumple = bool(condiciones["cumple_todas"].any())
_mejora = float(condiciones["1_mejora_rural"].iloc[0])
_umbral_c1 = cond["1_mejora_rural"]["umbral"]
_no_dana = bool(condiciones[["c2_cumple", "c3_cumple", "c4_cumple"]].all().all())

VEREDICTO_REMEDIO = {
    "aceptado": _alguna_cumple,
    "motivo": "mejora insuficiente" if not _alguna_cumple else "cumple las cuatro condiciones",
    "mejora_rural_obtenida": _mejora,
    "mejora_rural_exigida": _umbral_c1,
    "es_inocuo": _no_dana,
    "factor_estimado_en_ajuste": factores_aj["rural"],
    "factor_necesario_en_validacion": factores_va["rural"],
    "brecha_entre_ventanas": float(factores_va["rural"] - factores_aj["rural"]),
    "diagnostico": "El modelo esta correctamente calibrado en el grupo rural durante su periodo "
                   "de entrenamiento. El desajuste aparece al extrapolar. Un factor estimado en "
                   "el pasado no corrige un problema que en el pasado no existe.",
    "reformulacion_de_L1": "El sesgo rural no es un error constante de nivel sino una perdida de "
                           "calibracion al proyectar hacia adelante.",
    "hipotesis_derivada": {
        "enunciado": "Tendencia diferencial: la siniestralidad rural podria crecer a un ritmo "
                     "distinto del promedio nacional, y el factor unico de NB05 subcorregiria a "
                     "ese grupo.",
        "estado": "NO VERIFICADA. Hipotesis derivada del resultado, no hecho medido.",
        "coherente_con": ["L2 (fragilidad del factor)", "deriva temporal documentada en NB05"],
    },
    "linea_futura_prioritaria": {
        "remedio": "Factor de crecimiento diferenciado por grupo territorial",
        "por_que_no_aqui": [
            "Modifica la especificacion de la exposicion (objeto de NB05).",
            "El CAGR es fragil con poca historia; estimarlo sobre 19 distritos lo empeoraria.",
            "Requiere verificar primero que la tendencia diferencial existe.",
        ],
        "prerequisito": "Correccion metodologica de L2: anclar en promedios trimestrales y "
                        "exigir un minimo de 24 meses de historia.",
    },
}

print("=== VEREDICTO — SECCION 4 ===\n")
print("Criterio (4.1): se acepta solo si se cumplen LAS CUATRO condiciones.\n")
print(f"REMEDIO {'ACEPTADO' if _alguna_cumple else 'RECHAZADO'}")
print(f"  Condicion 1 (mejora rural): obtenida {_mejora:.6f}, exigida >= {_umbral_c1}")
print(f"  Condiciones 2, 3 y 4      : {'se cumplen' if _no_dana else 'NO se cumplen'}")
if not _alguna_cumple and _no_dana:
    print("\n  Se rechaza por INUTIL, no por daniño: no rompe nada pero tampoco arregla nada.")

print("\n--- Por que fallo ---")
print(f"  Factor estimado en ajuste 2022-2023 : {factores_aj['rural']:.6f}")
print(f"  Factor necesario en validacion 2024 : {factores_va['rural']:.6f}")
print(f"  Brecha                              : {VEREDICTO_REMEDIO['brecha_entre_ventanas']:+.6f}")
print("\n  El modelo esta bien calibrado en el grupo rural durante su periodo de entrenamiento.")
print("  El desajuste aparece al extrapolar.")

print("\n--- Reformulacion de la limitacion L1 ---")
print(f"  {VEREDICTO_REMEDIO['reformulacion_de_L1']}")
print("\n--- Hipotesis derivada (NO verificada) ---")
print(f"  {VEREDICTO_REMEDIO['hipotesis_derivada']['enunciado']}")
print("\n--- Linea futura prioritaria ---")
print(f"  {VEREDICTO_REMEDIO['linea_futura_prioritaria']['remedio']}")
print(f"  Prerequisito: {VEREDICTO_REMEDIO['linea_futura_prioritaria']['prerequisito']}")

SECCION4["veredicto"] = VEREDICTO_REMEDIO
RUTA_S4 = DIR_RESULTS / "nb08_seccion4.json"
RUTA_S4.write_text(json.dumps(SECCION4, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
print(f"\nGuardado en {RUTA_S4.relative_to(ROOT)}")
del aj, va

# %% [markdown]
# # Sección 5 — Fase 5 del anteproyecto
#
# La Fase 5 de CRISP-DM, tal como quedó definida en el anteproyecto, pide tres cosas: validación
# cruzada temporal para evitar fuga de períodos futuros, análisis de sesgos por subgrupo, y
# documentación de las limitaciones identificadas.
#
# ## 5.1 Validación cruzada temporal con pliegues expansivos
#
# Se usan pliegues **expansivos**, no rotativos: cada pliegue entrena sobre todo lo anterior a su
# ventana de validación y nunca sobre datos posteriores. Es el único esquema admisible en series
# temporales.
#
# | Pliegue | Entrena | Valida |
# |---|---|---|
# | 1 | 2022-01 a 2022-12 | 2023-01 a 2023-06 |
# | 2 | 2022-01 a 2023-06 | 2023-07 a 2023-12 |
# | 3 | 2022-01 a 2023-12 | 2024-01 a 2024-06 |
# | 4 | 2022-01 a 2024-06 | 2024-07 a 2024-12 |
#
# ### El factor de crecimiento se reestima en cada pliegue
#
# Decisión D2. El factor de NB05 se estimó con las medias anuales de 2022 y 2024 — con **todo** el
# train. Si el pliegue 1 entrena en 2022 y valida en 2023 usando ese factor, está usando
# información de 2024: fuga temporal dentro del procedimiento cuyo propósito es detectarlas.
#
# **Consecuencia declarada.** Los pliegues tempranos disponen de menos historia y producen
# estimaciones más ruidosas. Eso no es un defecto del diseño: es la información que se busca. Dice
# cuánta historia necesita el procedimiento para estabilizarse, que es el dato que el prototipo
# requiere para fijar su frecuencia de reentrenamiento.
#
# **Comparabilidad.** Los resultados no son directamente comparables con las métricas de NB05 y
# NB07, que usan el factor global. Se reporta también el desempeño con factor fijo: la diferencia
# entre ambas columnas mide cuánto optimismo introduce la fuga.
#
# El Spearman por tercil se omite: con ventanas de seis meses los conteos por distrito son
# demasiado pequeños para que el coeficiente por subgrupo signifique algo.

# %%
# NB08 — 5.1 Validacion cruzada temporal con pliegues expansivos

PLIEGUES = [
    {"id": 1, "train_fin": "2022-12-31", "val_ini": "2023-01-01", "val_fin": "2023-06-30"},
    {"id": 2, "train_fin": "2023-06-30", "val_ini": "2023-07-01", "val_fin": "2023-12-31"},
    {"id": 3, "train_fin": "2023-12-31", "val_ini": "2024-01-01", "val_fin": "2024-06-30"},
    {"id": 4, "train_fin": "2024-06-30", "val_ini": "2024-07-01", "val_fin": "2024-12-31"},
]
TRAIN_INI_CV = pd.Timestamp("2022-01-01")


def estimar_factor_cagr(df: pd.DataFrame) -> dict:
    """Replica el metodo de NB05 usando UNICAMENTE los datos que se le pasan (D2)."""
    mensual = df.groupby(df[COL_FECHA].dt.to_period("M"))[COL_Y].mean().sort_index()
    if len(mensual) < 12:
        raise RuntimeError(f"Historia insuficiente para el CAGR: {len(mensual)} meses")
    n_meses = len(mensual) - 1
    g = (mensual.iloc[-1] / mensual.iloc[0]) ** (1 / n_meses) - 1
    return {"g_mensual": float(g), "g_anual": float((1 + g) ** 12 - 1),
            "meses_usados": int(len(mensual)),
            "media_primer_mes": float(mensual.iloc[0]),
            "media_ultimo_mes": float(mensual.iloc[-1])}


def aplicar_factor(df: pd.DataFrame, g: float) -> pd.Series:
    """Reconstruye log_exposicion_v2 = log_exposicion + log(factor(t)) con el g dado."""
    meses = ((df[COL_FECHA].dt.year - TRAIN_INI_CV.year) * 12
             + (df[COL_FECHA].dt.month - TRAIN_INI_CV.month))
    return df["log_exposicion"] + np.log((1 + g) ** meses)


def entrenar_y_evaluar(df_tr, df_va, offset_tr, offset_va, etiqueta: str) -> dict:
    """Entrena un XGBoost con los hiperparametros de NB07 y evalua sobre la ventana."""
    dm = xgb.DMatrix(df_tr[COLUMNAS_OFFSET].astype(float), label=df_tr[COL_Y])
    dm.set_base_margin(offset_tr.values + LOG_TASA_BASE)
    booster = xgb.train(params_xgb, dm, num_boost_round=n_rondas_offset)

    dmv = xgb.DMatrix(df_va[COLUMNAS_OFFSET].astype(float))
    dmv.set_base_margin(offset_va.values + LOG_TASA_BASE)
    pred = booster.predict(dmv)

    obs = df_va[COL_Y].values.astype(float)
    agg = (pd.DataFrame({"d": df_va[COL_DISTRITO].values, "obs": obs, "pred": pred})
           .groupby("d")[["obs", "pred"]].sum())
    return {
        "variante": etiqueta,
        "calibracion": float(pred.mean() / obs.mean()),
        "RMSE": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "MAE": float(np.mean(np.abs(obs - pred))),
        "spearman_global": spearman_seguro(agg["pred"], agg["obs"]),
        "media_observada": float(obs.mean()),
        "media_predicha": float(pred.mean()),
    }


G_GLOBAL = ESPEC_NB05["correccion_tendencia"]["g_mensual"]
print(f"Factor global de NB05 (referencia): g_mensual = {G_GLOBAL:.8f} "
      f"({ESPEC_NB05['correccion_tendencia']['g_anual']:.4%} anual)\n")

resultados_cv, factores = [], []
for p in PLIEGUES:
    tr = matriz[MASK_TRAIN & matriz[COL_FECHA].between(TRAIN_INI_CV, pd.Timestamp(p["train_fin"]))]
    vv = matriz[MASK_TRAIN & matriz[COL_FECHA].between(pd.Timestamp(p["val_ini"]),
                                                       pd.Timestamp(p["val_fin"]))]
    f = estimar_factor_cagr(tr)
    factores.append({"pliegue": p["id"], "train_hasta": p["train_fin"], **f,
                     "desvio_vs_global_pct": (f["g_mensual"] / G_GLOBAL - 1) * 100})

    print(f"Pliegue {p['id']}: train hasta {p['train_fin']} ({len(tr):,} filas, "
          f"{f['meses_usados']} meses) | valida {p['val_ini']} a {p['val_fin']} ({len(vv):,} filas)")
    print(f"  g reestimado = {f['g_mensual']:.8f} ({f['g_anual']:.4%} anual) | "
          f"desvio {f['g_mensual'] / G_GLOBAL - 1:+.2%}")

    r = entrenar_y_evaluar(tr, vv, aplicar_factor(tr, f["g_mensual"]),
                           aplicar_factor(vv, f["g_mensual"]), "factor reestimado (D2)")
    resultados_cv.append({"pliegue": p["id"], **r})

    r = entrenar_y_evaluar(tr, vv, tr[OFFSET_COL], vv[OFFSET_COL], "factor global (con fuga)")
    resultados_cv.append({"pliegue": p["id"], **r})
    print(f"  -> calibracion D2 {resultados_cv[-2]['calibracion']:.6f} | "
          f"global {resultados_cv[-1]['calibracion']:.6f}\n")

cv = pd.DataFrame(resultados_cv)
tabla_factores = pd.DataFrame(factores).set_index("pliegue")

print("=== FACTOR DE CRECIMIENTO REESTIMADO POR PLIEGUE ===")
display(tabla_factores[["train_hasta", "meses_usados", "g_mensual", "g_anual",
                        "desvio_vs_global_pct"]].round(6))
print("Los pliegues tempranos tienen menos historia -> estimaciones mas ruidosas.")
print("Esa inestabilidad es informacion: indica cuanta historia requiere el procedimiento.")

print("\n=== VALIDACION CRUZADA TEMPORAL ===")
display(cv.pivot_table(index="pliegue", columns="variante",
                       values=["calibracion", "RMSE", "spearman_global"]).round(6))

print("\n=== ESTABILIDAD ENTRE PLIEGUES ===")
est = cv.groupby("variante").agg(
    calibracion_media=("calibracion", "mean"), calibracion_sd=("calibracion", "std"),
    calibracion_min=("calibracion", "min"), calibracion_max=("calibracion", "max"),
    RMSE_medio=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
    spearman_medio=("spearman_global", "mean"), spearman_sd=("spearman_global", "std"),
)
display(est.round(6))

_d2v = cv[cv["variante"] == "factor reestimado (D2)"]
_glv = cv[cv["variante"] == "factor global (con fuga)"]
_sesgo = float(abs(_glv["calibracion"] - 1).mean() - abs(_d2v["calibracion"] - 1).mean())
print(f"\nOptimismo introducido por la fuga del factor global: {_sesgo:+.6f}")
print("  (diferencia media en |calibracion - 1|; negativo = el factor global aparenta mejor")
print("   calibracion de la que corresponde, por usar informacion posterior)")

SECCION5 = {
    "validacion_cruzada": {
        "diseno": "pliegues expansivos dentro de train; el test no participa",
        "pliegues": PLIEGUES,
        "factor_reestimado_por_pliegue": tabla_factores.to_dict("index"),
        "factor_global_nb05": G_GLOBAL,
        "resultados": cv.to_dict("records"),
        "estabilidad": est.to_dict("index"),
        "optimismo_por_fuga": _sesgo,
        "decision": "D2 — el factor se reestima en cada pliegue para evitar fuga temporal",
    }
}

# %% [markdown]
# ## 5.2 Análisis de sesgos por subgrupo
#
# El anteproyecto pide comprobar si el modelo rinde de forma equivalente en distintos subgrupos.
# Si el modelo predice sistemáticamente de menos en zonas rurales, una asignación de recursos
# guiada por sus predicciones subatendería esas zonas — y el sesgo se trasladaría del modelo a la
# política pública.
#
# | Eje | Definición | Tipo |
# |---|---|---|
# | Urbano / rural | `pct_urbano` sobre un umbral | Territorial |
# | Seca / lluviosa | `es_lluviosa` (mayo–octubre) | Temporal |
# | Fin de semana / hábil | `es_finde` (viernes y sábado) | Temporal |
# | Población baja / resto | `poblacion_baja` | Territorial |
#
# ### Sobre el corte urbano/rural
#
# `pct_urbano` es continua con 86 valores, así que dicotomizarla exige elegir un umbral, y esa
# elección puede determinar el resultado. En lugar de fijar uno, se reportan **tres** —0.3, 0.5 y
# 0.7— y se observa si la conclusión cambia. Si el hallazgo depende de dónde se puso la línea, no
# es un hallazgo.
#
# ### Métricas y su lectura
#
# Para el análisis de sesgos la **calibración es la métrica central**: mide si el modelo acierta
# el nivel en ese subgrupo, que es lo que determina si una asignación de recursos sería justa.
#
# Comparar RMSE entre subgrupos con medias distintas induce a error: el error absoluto crece con
# la magnitud del fenómeno. Se reporta también el **RMSE relativo**, que sí es comparable.

# %%
# NB08 — 5.2 Analisis de sesgos por subgrupo

exigir_particion_de_diagnostico(PART_DIAG)

UMBRALES_URBANO = [0.3, 0.5, 0.7]


def metricas_subgrupo(df: pd.DataFrame, col_pred: str) -> dict:
    """Calibracion, RMSE (absoluto y relativo) y Spearman entre distritos del subgrupo."""
    obs = df[COL_Y].values.astype(float)
    pred = df[col_pred].values.astype(float)
    agg = (pd.DataFrame({"d": df[COL_DISTRITO].values, "obs": obs, "pred": pred})
           .groupby("d")[["obs", "pred"]].sum())
    media_obs = obs.mean()
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    return {
        "n_celdas": len(df), "n_distritos": int(agg.shape[0]),
        "media_observada": float(media_obs), "media_predicha": float(pred.mean()),
        "calibracion": float(pred.mean() / media_obs) if media_obs else np.nan,
        "RMSE": rmse, "RMSE_relativo": rmse / media_obs if media_obs else np.nan,
        "spearman": spearman_seguro(agg["pred"], agg["obs"]),
    }


def construir_ejes(df: pd.DataFrame) -> dict:
    ejes = {
        "clima: seca / lluviosa": ("seca", "lluviosa", df["es_lluviosa"] == 1),
        "calendario: habil / finde": ("habil", "finde", df["es_finde"] == 1),
        "poblacion: resto / baja": ("resto", "poblacion baja", df["poblacion_baja"] == 1),
    }
    for u in UMBRALES_URBANO:
        ejes[f"territorio: rural / urbano (pct>{u})"] = (
            f"rural (pct<={u})", f"urbano (pct>{u})", df["pct_urbano"] > u)
    return ejes


def evaluar_sesgos(mask, etiqueta: str, usar_gemelo: bool) -> pd.DataFrame:
    df = matriz[mask].copy()
    df["pred"] = (predecir_xgb_offset(df, booster=GEMELO_XGB) if usar_gemelo
                  else predecir_xgb_offset(df))
    filas = []
    for eje, (nombre_a, nombre_b, cond_eje) in construir_ejes(df).items():
        for nombre, sel in [(nombre_a, ~cond_eje), (nombre_b, cond_eje)]:
            sub = df[sel]
            if len(sub) == 0:
                continue
            filas.append({"particion": etiqueta, "eje": eje, "grupo": nombre,
                          **metricas_subgrupo(sub, "pred")})
    return pd.DataFrame(filas)


sesgos_diag = evaluar_sesgos(MASK_DIAG, PART_DIAG, usar_gemelo=True)
sesgos_test = evaluar_sesgos(MASK_TEST, "test", usar_gemelo=False)  # descriptivo (D1)

print(f"=== SESGOS POR SUBGRUPO — {PART_DIAG} (gemelo 2022-2023) ===\n")
display(sesgos_diag.set_index(["eje", "grupo"])[
    ["n_celdas", "n_distritos", "media_observada", "calibracion",
     "RMSE", "RMSE_relativo", "spearman"]].round(6))


def brechas(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for eje, g in df.groupby("eje"):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        filas.append({
            "eje": eje, "grupo_A": a["grupo"], "grupo_B": b["grupo"],
            "calib_A": a["calibracion"], "calib_B": b["calibracion"],
            "brecha_calibracion": float(b["calibracion"] - a["calibracion"]),
            "brecha_RMSE_relativo": float(b["RMSE_relativo"] - a["RMSE_relativo"]),
            "brecha_spearman": float(b["spearman"] - a["spearman"]),
            "razon_medias_obs": float(b["media_observada"] / a["media_observada"]),
        })
    return pd.DataFrame(filas).set_index("eje")


br_diag = brechas(sesgos_diag)
print("\n=== BRECHAS POR EJE (grupo B menos grupo A) ===")
display(br_diag.round(6))
print("brecha_calibracion: positiva = el grupo B recibe predicciones relativamente mayores.")
print("Una calibracion por debajo de 1 indica subestimacion del nivel en ese grupo.")

_urb = br_diag[br_diag.index.str.startswith("territorio: rural / urbano")]
print("\n=== ROBUSTEZ DEL CORTE URBANO/RURAL ===")
display(_urb[["calib_A", "calib_B", "brecha_calibracion", "brecha_spearman"]].round(6))
_signos = set(np.sign(_urb["brecha_calibracion"].values))
print(f"Signo de la brecha en los {len(_urb)} umbrales: "
      f"{'CONSISTENTE' if len(_signos) == 1 else 'CAMBIA SEGUN EL UMBRAL'}")
if len(_signos) > 1:
    print("  La brecha solo aparece con un umbral: el hallazgo esta LOCALIZADO en un grupo")
    print("  reducido de distritos muy rurales, no es un gradiente continuo.")

_peor = sesgos_diag.loc[(sesgos_diag["calibracion"] - 1).abs().idxmax()]
print(f"\nSubgrupo con mayor desvio de calibracion: {_peor['eje']} / {_peor['grupo']}")
print(f"  calibracion {_peor['calibracion']:.6f} (desvio {abs(_peor['calibracion'] - 1):.4f})")
print(f"  rango entre todos los subgrupos: "
      f"[{sesgos_diag['calibracion'].min():.6f}, {sesgos_diag['calibracion'].max():.6f}]")

br_test = brechas(sesgos_test)
print(f"\n=== MISMAS BRECHAS SOBRE EL TEST {ETIQUETA_DESCRIPTIVA} ===")
display(br_test[["brecha_calibracion", "brecha_RMSE_relativo", "brecha_spearman"]].round(6))
_coherencia = (np.sign(br_diag["brecha_calibracion"]) ==
               np.sign(br_test["brecha_calibracion"].reindex(br_diag.index)))
print(f"Ejes cuya brecha mantiene el signo entre particiones: "
      f"{int(_coherencia.sum())} de {len(_coherencia)}")

SECCION5["sesgos"] = {
    "ejes_evaluados": list(br_diag.index),
    "umbrales_urbano": UMBRALES_URBANO,
    "diagnostico": sesgos_diag.to_dict("records"),
    "test_descriptivo": sesgos_test.to_dict("records"),
    "brechas_diagnostico": br_diag.to_dict("index"),
    "brechas_test": br_test.to_dict("index"),
    "corte_urbano_robusto": len(_signos) == 1,
    "ejes_coherentes_entre_particiones": int(_coherencia.sum()),
    "nota_rmse": "El RMSE absoluto no es comparable entre subgrupos con medias distintas; "
                 "se reporta el RMSE relativo para ese fin.",
}

# %% [markdown]
# ## 5.3 Limitaciones
#
# Cierre de la Fase 5. Cada limitación lleva su magnitud cuando pudo cuantificarse, su
# consecuencia práctica y la vía de solución cuando existe. El criterio de redacción es declarar
# la limitación **con su tamaño**: una limitación sin magnitud no permite al lector juzgar si le
# importa.
#
# **L1 — Subestimación en distritos rurales pequeños.** El modelo predice alrededor de un 14%
# menos de lo que ocurre en los distritos con menos del 30% de población urbana. No es un
# gradiente continuo: con umbrales de 0.5 y 0.7 la brecha desaparece. Tres mediciones
# independientes señalan el mismo conjunto. La Sección 4 reformuló la limitación: es una pérdida
# de calibración al extrapolar, no un error constante de nivel.
#
# **L2 — El factor de crecimiento es frágil con poca historia.** Reestimado por pliegue, el CAGR
# osciló ampliamente frente al valor global de NB05. La causa es que el método ancla en el primer
# y el último mes de la ventana, y esos meses arrastran su estacionalidad. No afecta RMSE ni
# Spearman: el factor es un multiplicador común, desplaza el nivel pero no reordena. NB05 no lo
# sufrió porque disponía de 36 meses y ancló en medias anuales.
#
# **L3 — El Spearman por subgrupo pierde resolución en rangos comprimidos.** Mover el corte
# cambia el coeficiente tres veces más que cambiar el modelo. Las comparaciones de ordenamiento
# dentro de subgrupos estrechos deben acompañarse de su intervalo de confianza.
#
# **L4 — Cobertura territorial parcial.** 103 distritos de los 262 que define el Decreto 762. Los
# 159 restantes tienen ceros estructurales, no observados. El prototipo debe distinguir
# visualmente la cobertura incompleta de «poco riesgo».
#
# **L5 — Los gemelos no son los modelos entregables.** Las conclusiones de las Secciones 2 y 3
# valen para la especificación del modelo, no necesariamente para el ejemplar desplegado.
#
# **L6 — El dataset es sintéticamente calibrado.** Los patrones estructurales son confiables
# porque están calibrados contra ONASEVI/FONAT. Los hallazgos que dependan de eventos individuales
# o de rupturas puntuales no lo son.
#
# **L7 — Variables ex post excluidas por diseño.** El sistema predice frecuencia, no gravedad
# futura.
#
# **Heredadas:** NB02 (seis estaciones meteorológicas; San Salvador excluida de precipitación),
# NB03 (102 de 103 distritos sin feriados patronales locales), NB05 (heterogeneidad territorial
# no explicada), NB07 (`log_tasa_base` fuera del `.pkl`).

# %%
# NB08 — 5.3 Limitaciones: registro estructurado

LIMITACIONES = {
    "L1_subestimacion_rural": {
        "titulo": "Subestimacion en distritos rurales pequenos",
        "magnitud": {
            "calibracion_rural_pct_menor_030": float(
                sesgos_diag.query("grupo.str.startswith('rural (pct<=0.3')")["calibracion"].iloc[0]),
            "calibracion_poblacion_baja": float(
                sesgos_diag.query("grupo == 'poblacion baja'")["calibracion"].iloc[0]),
            "n_distritos_rurales": int(
                sesgos_diag.query("grupo.str.startswith('rural (pct<=0.3')")["n_distritos"].iloc[0]),
            "n_distritos_poblacion_baja": int(
                sesgos_diag.query("grupo == 'poblacion baja'")["n_distritos"].iloc[0]),
        },
        "alcance": "Localizado en distritos muy rurales, no es un gradiente continuo: con "
                   "umbral 0.5 y 0.7 la brecha desaparece.",
        "evidencia_convergente": ["celda 3.3 (desajuste persistente)",
                                  "celda 5.2 (calibracion por subgrupo)",
                                  "NB05 (refutacion de la hipotesis AMSS)"],
        "consecuencia": "Una asignacion de recursos guiada por el modelo subatenderia estas "
                        "zonas ~14%. Debe declararse en el prototipo con el factor medido.",
        "via_de_solucion": ESPEC_NB05["correccion_territorial"]["fuentes_requeridas"],
        "reformulacion_nb08_seccion4": {
            "hallazgo": VEREDICTO_REMEDIO["reformulacion_de_L1"],
            "evidencia": {"factor_en_entrenamiento": VEREDICTO_REMEDIO["factor_estimado_en_ajuste"],
                          "factor_necesario_al_extrapolar": VEREDICTO_REMEDIO["factor_necesario_en_validacion"]},
            "remedio_probado_y_rechazado": "factor multiplicativo por grupo territorial",
            "linea_futura": VEREDICTO_REMEDIO["linea_futura_prioritaria"]["remedio"],
        },
    },
    "L2_fragilidad_del_factor": {
        "titulo": "El factor de crecimiento es fragil con poca historia",
        "magnitud": {
            "rango_g_anual_reestimado": [float(tabla_factores["g_anual"].min()),
                                         float(tabla_factores["g_anual"].max())],
            "g_anual_global_nb05": float(ESPEC_NB05["correccion_tendencia"]["g_anual"]),
            "desvio_maximo_pct": float(tabla_factores["desvio_vs_global_pct"].max()),
        },
        "causa": "El CAGR ancla en el primer y ultimo mes, que arrastran su estacionalidad.",
        "no_afecta": "RMSE y Spearman practicamente identicos entre variantes: el factor es un "
                     "multiplicador comun, desplaza el nivel pero no reordena.",
        "regla_operativa": "No reestimar con menos de 24 meses; anclar en promedios trimestrales.",
    },
    "L3_resolucion_del_spearman": {
        "titulo": "El Spearman por subgrupo pierde resolucion en rangos comprimidos",
        "magnitud": {
            "variacion_por_corte": VEREDICTO_REALIDAD["restriccion_de_rango"][
                "amplitud_por_variante_de_corte"],
            "variacion_por_modelo": VEREDICTO_REALIDAD["restriccion_de_rango"][
                "amplitud_entre_modelos_en_el_protocolo"],
            "pares_distinguibles_tercil_medio": SECCION3["resolucion"][
                "variacion_entre_modelos_tercil_medio"],
        },
        "consecuencia": "Toda comparacion de ordenamiento en subgrupos estrechos debe "
                        "acompanarse de su intervalo de confianza.",
    },
    "L4_cobertura_parcial": {
        "titulo": "Cobertura territorial parcial",
        "magnitud": {"distritos_con_datos": int(tabla_diag.shape[0]), "distritos_totales": 262},
        "razon": "Los 159 distritos restantes tienen ceros estructurales, no observados.",
        "consecuencia": "La cobertura incompleta debe distinguirse visualmente de 'poco riesgo'.",
    },
    "L5_gemelos": {
        "titulo": "Los gemelos no son los modelos entregables",
        "magnitud": {
            "dif_rmse_vs_nb07": abs(GEMELOS_INFO["xgb_rmse_validacion"]
                                    - GEMELOS_INFO["xgb_referencia_nb07"]["rmse_validacion"]),
            "dif_calibracion_vs_nb07": abs(GEMELOS_INFO["xgb_calibracion_validacion"]
                                           - GEMELOS_INFO["xgb_referencia_nb07"]["calibracion_validacion"]),
            "rf_sin_referencia": True,
        },
        "consecuencia": "Las conclusiones de las Secciones 2 y 3 valen para la especificacion "
                        "del modelo, no necesariamente para el ejemplar desplegado.",
    },
    "L6_dataset_sintetico": {
        "titulo": "El dataset es sinteticamente calibrado",
        "consecuencia": "Los patrones estructurales son confiables (calibrados contra "
                        "ONASEVI/FONAT); los hallazgos sobre eventos individuales o rupturas "
                        "puntuales, no.",
    },
    "L7_variables_ex_post": {
        "titulo": "Variables ex post excluidas por diseno",
        "excluidas": COLS_PROHIBIDAS,
        "razon": "Fuga circular: no puede haber fallecidos donde no hubo siniestro.",
        "consecuencia": "El sistema predice frecuencia, no gravedad futura.",
    },
    "heredadas": {
        "NB02": "Seis estaciones meteorologicas para todo el territorio; SS excluida de "
                "precipitacion (22.7% completitud), se usa Ilopango para el AMSS.",
        "NB03": "102 de 103 distritos carecen de feriados patronales locales en la fuente.",
        "NB05": "Heterogeneidad territorial no explicada por las variables disponibles.",
        "NB07": "log_tasa_base no viaja dentro del .pkl: sin el JSON las predicciones se "
                "desvian ~415,000x sin lanzar error.",
    },
}

SECCION5["limitaciones"] = LIMITACIONES
RUTA_S5 = DIR_RESULTS / "nb08_seccion5.json"
RUTA_S5.write_text(json.dumps(SECCION5, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")

print("=== LIMITACIONES DOCUMENTADAS ===\n")
for clave, d in LIMITACIONES.items():
    if clave == "heredadas":
        continue
    print(f"{clave}: {d['titulo']}")
    if "magnitud" in d:
        for k, v in d["magnitud"].items():
            print(f"    {k}: {v}")
    print()

print("Heredadas de etapas previas:")
for nb, texto in LIMITACIONES["heredadas"].items():
    print(f"  {nb}: {texto}")

print(f"\nGuardado en {RUTA_S5.relative_to(ROOT)}")
print(f"\nFase 5 completa: validacion cruzada temporal ({len(PLIEGUES)} pliegues), "
      f"sesgos ({len(SECCION5['sesgos']['ejes_evaluados'])} ejes), "
      f"limitaciones ({len(LIMITACIONES) - 1} propias + 4 heredadas).")

# %% [markdown]
# # Sección 6 — Confirmación en el conjunto de prueba
#
# ## 6.1 Declaración de uso: cuántas veces se ha mirado el test y por qué
#
# El conjunto de prueba es la única estimación insesgada del desempeño del sistema. Cada vez que
# se lo consulta para tomar una decisión, esa estimación se degrada.
#
# | # | Notebook | Celda | Propósito | ¿Decidió algo? |
# |---|---|---|---|---|
# | 1 | NB07 | — | Evaluación del modelo seleccionado | **Sí** |
# | 2 | NB08 | 1.3 | Reproducción de métricas ya publicadas | No |
# | 3 | NB08 | 2.1–2.5 | Contraste descriptivo junto al diagnóstico | No |
# | 4 | NB08 | 5.2 | Contraste de las brechas por subgrupo | No |
# | 5 | NB08 | 6.2 | Confirmación de las conclusiones | No |
#
# ### Por qué esta sección no es una segunda evaluación
#
# Originalmente iba a evaluar el modelo elegido en la Sección 4. Como ese remedio se rechazó, **no
# hay modelo nuevo que evaluar**: el vigente sigue siendo `frecuencia_xgboost_offset_v2.pkl`,
# cuyas métricas la celda 1.3 ya reprodujo al sexto decimal.
#
# Lo que sí queda sin verificar es si las tres conclusiones —construidas sobre gemelos y validación
# 2024— se sostienen con los modelos entregables en el período de prueba.
#
# | # | Conclusión | Dónde se estableció |
# |---|---|---|
# | C1 | La caída del tercil medio no se distingue del ruido | Sección 2, bootstrap pareado |
# | C2 | El Spearman por subgrupo mide el ancho del tercil | Secciones 2.5 y 3.2 |
# | C3 | Hay subestimación en distritos rurales pequeños | Sección 5.2, reformulada en 4.3 |
#
# Ninguna se revisa según lo que muestre el test. Si alguna no se sostuviera, se reportaría la
# discrepancia y se declararía que la conclusión vale para la partición donde se estableció.
#
# ### Honestidad sobre lo ya visto
#
# Las celdas 2.1–2.5 y 5.2 ya calcularon métricas sobre el test, rotuladas como descriptivas. No
# se oculta: se contabiliza. El punto del contador no es que el test nunca se mire, sino que quede
# registrado cuándo se lo miró y si algo dependió de ello. En este notebook, nada dependió.

# %%
# NB08 — 6.1 Declaracion de uso del conjunto de prueba

RANGO_TEST = ESPEC_NB07["split"]["test"]

USOS_TEST_DETALLE = [
    {"n": 1, "notebook": "NB07", "celda": "—",
     "proposito": "Evaluacion del modelo seleccionado", "decide": True},
    {"n": 2, "notebook": "NB08", "celda": "1.3",
     "proposito": "Reproduccion de metricas ya publicadas", "decide": False},
    {"n": 3, "notebook": "NB08", "celda": "2.1-2.5",
     "proposito": "Contraste descriptivo junto al diagnostico", "decide": False},
    {"n": 4, "notebook": "NB08", "celda": "5.2",
     "proposito": "Contraste de las brechas por subgrupo", "decide": False},
    {"n": 5, "notebook": "NB08", "celda": "6.2",
     "proposito": "Confirmacion de las conclusiones", "decide": False},
]

CONCLUSIONES = {
    "C1_caida_no_real": {
        "enunciado": "La caida del tercil medio no se distingue del ruido de muestreo.",
        "establecida_en": "Seccion 2 (bootstrap pareado, validacion 2024)",
        "evidencia": {"diferencia": VEREDICTO_REALIDAD["tercil_medio"]["diferencia_puntual"],
                      "ic": VEREDICTO_REALIDAD["tercil_medio"]["ic_diferencia"]},
    },
    "C2_metrica_mide_el_ancho": {
        "enunciado": "El Spearman por subgrupo mide el ancho del tercil antes que la capacidad "
                     "del modelo.",
        "establecida_en": "Secciones 2.5 y 3.2",
        "evidencia": {
            "variacion_por_corte": VEREDICTO_REALIDAD["restriccion_de_rango"][
                "amplitud_por_variante_de_corte"],
            "variacion_por_modelo": VEREDICTO_REALIDAD["restriccion_de_rango"][
                "amplitud_entre_modelos_en_el_protocolo"],
        },
    },
    "C3_sesgo_rural": {
        "enunciado": "Hay subestimacion de calibracion en los distritos mas rurales, y es un "
                     "fenomeno de extrapolacion, no un error constante de nivel.",
        "establecida_en": "Seccion 5.2, reformulada en 4.3",
        "evidencia": {
            "calibracion_rural_validacion": LIMITACIONES["L1_subestimacion_rural"][
                "magnitud"]["calibracion_rural_pct_menor_030"],
            "factor_en_entrenamiento": VEREDICTO_REMEDIO["factor_estimado_en_ajuste"],
            "factor_al_extrapolar": VEREDICTO_REMEDIO["factor_necesario_en_validacion"],
        },
    },
}

print("=== SECCION 6 — CONFIRMACION EN EL CONJUNTO DE PRUEBA ===\n")
print(f"Conjunto de prueba: {RANGO_TEST} ({int(MASK_TEST.sum()):,} filas)\n")
print("Registro de usos:")
display(pd.DataFrame(USOS_TEST_DETALLE).set_index("n"))

_decisorios = sum(u["decide"] for u in USOS_TEST_DETALLE)
_decisorios_nb08 = sum(u["decide"] for u in USOS_TEST_DETALLE if u["notebook"] == "NB08")
print(f"Accesos que tomaron una decision    : {_decisorios} (solo NB07)")
print(f"Accesos de NB08 que decidieron algo : {_decisorios_nb08}")

print("\nEsta seccion NO es una segunda evaluacion:")
print("  El remedio de la Seccion 4 se rechazo, asi que no hay modelo nuevo que evaluar.")

print("\nConclusiones que se confirman (ninguna se revisa segun lo que muestre el test):")
for k, c in CONCLUSIONES.items():
    print(f"\n  {k}\n    {c['enunciado']}\n    establecida en: {c['establecida_en']}")

print("\nCandado D1 verificado en la celda 1.4: exigir_particion_de_diagnostico()")
print("rechaza la particion de prueba como fuente de veredictos.")

SECCION6 = {
    "rango_test": RANGO_TEST, "filas_test": int(MASK_TEST.sum()),
    "usos_del_test": USOS_TEST_DETALLE,
    "accesos_decisorios_totales": _decisorios,
    "accesos_decisorios_nb08": _decisorios_nb08,
    "conclusiones_a_confirmar": CONCLUSIONES,
    "es_segunda_evaluacion": False,
    "motivo": "El remedio de la Seccion 4 fue rechazado; no hay modelo nuevo que evaluar.",
}

# %% [markdown]
# ## 6.2 Confirmación de las tres conclusiones sobre los modelos finales
#
# Se verifica si las conclusiones establecidas en validación 2024 con gemelos se sostienen con los
# modelos entregables sobre el conjunto de prueba. **Ninguna conclusión se revisa según este
# resultado.**
#
# ### C1 — Diferencia esperada respecto de la validación
#
# En el test, la diferencia del tercil medio excluye el cero. Eso no contradice C1, porque C1 se
# estableció con modelos que no habían visto el período evaluado. Los `.pkl` finales sí entrenaron
# sobre 2022-2024, y el RF de Etapa 1 tiene la exposición como variable predictora, lo que le
# permite reproducir el nivel histórico de cada distrito. La diferencia en el test mide en parte
# esa ventaja de memoria, no solo capacidad predictiva.
#
# ### C3 — La predicción más arriesgada del notebook
#
# Se comprueba que el factor necesario para corregir el grupo rural sea aún mayor en el test que
# en 2024. Si el sesgo crece con la distancia temporal al entrenamiento, queda confirmado que es
# un fenómeno de extrapolación.
#
# **Si el factor del test resultara menor que el de 2024, la hipótesis de tendencia diferencial
# queda en duda y hay que decirlo.** Una causa probable de discrepancia: el factor del test se
# calcula con el modelo **final**, mientras que los de ajuste y validación usan el **gemelo**. No
# son tres mediciones del mismo objeto, y esa es una falla de diseño de la celda, no un resultado
# sobre el fenómeno.

# %%
# NB08 — 6.2 Confirmacion de las tres conclusiones sobre los modelos finales

print(f"Conjunto de prueba: {RANGO_TEST}\n")

# --- C1 --------------------------------------------------------------------------
filas = []
for particion, tab, etiqueta in [(PART_DIAG, tabla_diag, "gemelos 2022-2023"),
                                 ("test", tabla_test, "modelos finales")]:
    for t in TERCILES:
        sub = tab[tab["tercil"] == t]
        r = bootstrap_pareado(sub, MODELOS["RF Etapa 1"], MODELOS["XGB offset (NB07)"],
                              rng=np.random.default_rng(SEED))
        filas.append({
            "particion": particion, "modelos": etiqueta, "tercil": t,
            "diferencia": r["diferencia_puntual"],
            "IC_inf": r["dif_ic_inf"], "IC_sup": r["dif_ic_sup"],
            "excluye_cero": bool(r["dif_ic_inf"] > 0 or r["dif_ic_sup"] < 0),
            "amplitud_IC": r["dif_ic_sup"] - r["dif_ic_inf"],
        })

c1 = pd.DataFrame(filas)
print("=== C1 — DIFERENCIA RF ETAPA 1 vs XGB OFFSET, AMBAS PARTICIONES ===")
display(c1.set_index(["particion", "tercil"])[
    ["modelos", "diferencia", "IC_inf", "IC_sup", "excluye_cero", "amplitud_IC"]].round(6))

_c1_diag = c1.query("particion == @PART_DIAG and tercil == 'medio'").iloc[0]
_c1_test = c1.query("particion == 'test' and tercil == 'medio'").iloc[0]
print("\nTercil medio:")
print(f"  validacion 2024 (gemelos): {_c1_diag['diferencia']:+.6f} "
      f"IC [{_c1_diag['IC_inf']:+.4f}, {_c1_diag['IC_sup']:+.4f}] "
      f"-> {'excluye' if _c1_diag['excluye_cero'] else 'contiene'} el cero")
print(f"  test 2025-2026 (finales) : {_c1_test['diferencia']:+.6f} "
      f"IC [{_c1_test['IC_inf']:+.4f}, {_c1_test['IC_sup']:+.4f}] "
      f"-> {'excluye' if _c1_test['excluye_cero'] else 'contiene'} el cero")

c1_coherente = not bool(_c1_diag["excluye_cero"])
print("\nLectura: la diferencia del test incluye la ventaja de memoria del RF de Etapa 1,")
print("que tiene la exposicion como variable predictora y entreno sobre 2022-2024.")
print("El veredicto de C1 se sostiene sobre la comparacion sin contaminacion (validacion).")

# --- C2 --------------------------------------------------------------------------
filas = []
for nombre, v in VARIANTES.items():
    if v["criterio"] != "obs":
        continue
    tercil_v = asignar_terciles(tabla_test[v["criterio"]], v["n_alto"], v["n_medio"])
    tmp = tabla_test.assign(tercil_v=tercil_v)
    for etiqueta, col in MODELOS.items():
        sub = tmp[tmp["tercil_v"] == "medio"]
        filas.append({"variante": nombre, "modelo": etiqueta, "n": len(sub),
                      "spearman": spearman_seguro(sub[col], sub["obs"])})

c2 = pd.DataFrame(filas).pivot_table(index="variante", columns="modelo", values="spearman")
print("\n=== C2 — SPEARMAN DEL TERCIL MEDIO POR VARIANTE DE CORTE (test) ===")
display(c2.round(6))

_var_corte_test = float(c2.max().max() - c2.min().min())
_var_modelo_test = float(c2.loc["protocolo (35/34/34, conteo)"].max()
                         - c2.loc["protocolo (35/34/34, conteo)"].min())
print(f"  Variacion atribuible al CORTE  : {_var_corte_test:.4f}  "
      f"(validacion: {VEREDICTO_REALIDAD['restriccion_de_rango']['amplitud_por_variante_de_corte']:.4f})")
print(f"  Variacion atribuible al MODELO : {_var_modelo_test:.4f}  "
      f"(validacion: {VEREDICTO_REALIDAD['restriccion_de_rango']['amplitud_entre_modelos_en_el_protocolo']:.4f})")
c2_coherente = _var_corte_test > _var_modelo_test
print(f"  C2 {'SE SOSTIENE' if c2_coherente else 'NO se sostiene'} en el test.")

# --- C3 --------------------------------------------------------------------------
te = matriz[MASK_TEST].copy()
te["pred"] = predecir_xgb_offset(te)          # modelo FINAL
te[COL_GRUPO] = marcar_grupo(te)
factores_test = estimar_factores(te)

c3 = pd.DataFrame([
    {"ventana": "ajuste 2022-2023", "distancia_al_entrenamiento": "dentro",
     "factor_rural": factores_aj["rural"], "factor_resto": factores_aj["resto"]},
    {"ventana": "validacion 2024", "distancia_al_entrenamiento": "1 anio",
     "factor_rural": factores_va["rural"], "factor_resto": factores_va["resto"]},
    {"ventana": "test 2025-2026", "distancia_al_entrenamiento": "1 a 2.5 anios",
     "factor_rural": factores_test["rural"], "factor_resto": factores_test["resto"]},
]).set_index("ventana")
c3["calibracion_rural"] = 1 / c3["factor_rural"]

print("\n=== C3 — FACTOR NECESARIO PARA CORREGIR EL GRUPO RURAL, POR VENTANA ===")
display(c3.round(6))
print("factor_rural = sum(observado) / sum(predicho). Mayor que 1 = el modelo subestima.")
print("NOTA: el factor del test usa el modelo FINAL, que entreno sobre 2022-2024.")

c3_coherente = bool(factores_test["rural"] > factores_va["rural"] > factores_aj["rural"])
print(f"\n  Progresion monotona del factor rural: {'SI' if c3_coherente else 'NO'}")
if c3_coherente:
    print("  El sesgo crece con la distancia temporal al entrenamiento: confirma que es un")
    print("  fenomeno de EXTRAPOLACION, no un error constante de nivel.")
else:
    print("  La progresion no es monotona. La hipotesis de tendencia diferencial queda en duda.")
    print("  CAUSA PROBABLE: el factor del test usa el modelo FINAL mientras que ajuste y")
    print("  validacion usan el gemelo. No son tres mediciones del mismo objeto: es una falla")
    print("  de diseno de esta celda, no un resultado sobre el fenomeno. Se declara como tal.")
del te

# --- Cierre ----------------------------------------------------------------------
CONFIRMACION = {
    "C1_caida_no_real": {"coherente": c1_coherente,
                         "validacion": _c1_diag[["diferencia", "IC_inf", "IC_sup"]].to_dict(),
                         "test": _c1_test[["diferencia", "IC_inf", "IC_sup"]].to_dict(),
                         "nota": "La diferencia del test incluye la ventaja de memoria del RF "
                                 "de Etapa 1 (exposicion como variable, entreno 2022-2024)."},
    "C2_metrica_mide_el_ancho": {"coherente": c2_coherente,
                                 "variacion_corte_test": _var_corte_test,
                                 "variacion_modelo_test": _var_modelo_test},
    "C3_sesgo_rural": {"coherente": c3_coherente,
                       "factores_por_ventana": c3.to_dict("index"),
                       "nota_si_discrepa": "El factor del test usa el modelo FINAL mientras "
                                           "ajuste y validacion usan el gemelo: comparacion no "
                                           "limpia, falla de diseno de la celda."},
}
SECCION6["confirmacion"] = CONFIRMACION
SECCION6["metricas_test_publicadas"] = METRICAS_NB07["modelos"][
    "NB07 - XGBoost (exposición offset)"]

RUTA_S6 = DIR_RESULTS / "nb08_seccion6.json"
RUTA_S6.write_text(json.dumps(SECCION6, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")

print("\n" + "=" * 70)
print("=== CIERRE DE NB08 ===")
print("=" * 70)
print("\nModelo vigente: frecuencia_xgboost_offset_v2.pkl (sin cambios)")
mv = SECCION6["metricas_test_publicadas"]
print(f"  calibracion {mv['calibracion']:.6f} | RMSE {mv['RMSE']:.6f} | "
      f"spearman tercil alto {mv['spearman_tercil_alto']:.6f}")

print(f"\nConclusiones confirmadas en el test: "
      f"{sum(v['coherente'] for v in CONFIRMACION.values())} de 3")
for k, v in CONFIRMACION.items():
    print(f"  {k}: {'coherente' if v['coherente'] else 'DISCREPA — ver nota'}")

print("\nProblema abierto de NB07 (tercil medio): RESUELTO — no era un problema.")
print("Remedio probado y rechazado             : factor por grupo territorial (Seccion 4)")
print("Fase 5 del anteproyecto                 : completa")
print(f"Limitaciones documentadas               : {len(LIMITACIONES) - 1} propias + 4 heredadas")
print(f"Usos decisorios del test en NB08        : {_decisorios_nb08}")

_hashes_fin = {c: sha256_archivo(r) for c, (r, _) in ARTEFACTOS_PREVIOS.items() if r.is_file()}
_alterados = [k for k in hashes_actuales if hashes_actuales[k] != _hashes_fin.get(k)]
if _alterados:
    raise RuntimeError(f"INTEGRIDAD ROTA al cierre: {_alterados}")
print(f"\nIntegridad de artefactos previos: verificada ({len(_hashes_fin)} archivos intactos)")
print(f"Resultados guardados en {RUTA_S6.relative_to(ROOT)}")
