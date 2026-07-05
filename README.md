# Siniestralidad Vial El Salvador

Análisis de patrones espacio-temporales de accidentes de tránsito en El Salvador
mediante técnicas de Machine Learning: clustering espacio-temporal, modelos de
conteo (Poisson / Binomial Negativa) e interpretabilidad (SHAP).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Estado](https://img.shields.io/badge/estado-Etapa%201-yellow)
![Metodología](https://img.shields.io/badge/metodología-CRISP--DM-green)
![Seed](https://img.shields.io/badge/reproducibilidad-seed%2042-lightgrey)

> **Proyecto de Grado** — Escuela de Ingeniería en Sistemas Informáticos,
> Facultad de Ingeniería y Arquitectura, Universidad de El Salvador.

---

## Tabla de contenidos

- [Siniestralidad Vial El Salvador](#siniestralidad-vial-el-salvador)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Contexto](#contexto)
  - [Nota metodológica importante](#nota-metodológica-importante)
  - [Estructura del repositorio](#estructura-del-repositorio)
  - [Fuentes de datos](#fuentes-de-datos)
  - [Instalación](#instalación)
  - [Reproducción](#reproducción)
  - [Notebooks (Etapa 1)](#notebooks-etapa-1)
  - [Convenciones del proyecto](#convenciones-del-proyecto)
  - [Licencia](#licencia)

---

## Contexto

Los accidentes de tránsito son una problemática creciente en El Salvador: de
17 408 siniestros en 2022 se pasó a 22 265 en 2025, con más de 1 200 fallecidos
anuales (ONASEVI/FONAT). Este proyecto aplica un flujo analítico completo para
identificar **zonas y períodos de alto riesgo** y **predecir la frecuencia de
accidentes**, integrando variables climáticas, de exposición poblacional y de
calendario.

El trabajo sigue la metodología **CRISP-DM** y se organiza en cuatro etapas de
evaluación. Este repositorio corresponde a la **Etapa 1** (descripción de datos,
EDA, preprocesamiento, modelos baseline e interpretabilidad inicial).

Técnicas previstas a lo largo del proyecto:

- **Clustering espacio-temporal** — ST-DBSCAN
- **Modelos de conteo** — Regresión de Poisson y Binomial Negativa (selección por AIC)
- **Modelos predictivos** — Random Forest y XGBoost
- **Interpretabilidad** — SHAP

---

## Nota metodológica importante

El microdato oficial a nivel de incidente **no es de acceso público** en El
Salvador; el ONASEVI/FONAT solo publica agregados. Por ello, los archivos de
siniestralidad (`siniestros.csv`, `victimas.csv`) son un **conjunto sintético
calibrado a los agregados oficiales**: reproducen exactamente los totales
anuales oficiales (siniestros, fallecidos y lesionados 2022–2025) y preservan
las distribuciones documentadas del fenómeno.

El objetivo del proyecto es demostrar que la **tubería analítica recupera la
estructura conocida** de los datos, no generar hallazgos epidemiológicos nuevos.
Las fuentes de exposición (censo, clima, feriados) son **datos reales oficiales
o públicos**. Ver detalle completo en [`docs/fuentes_de_datos.md`](docs/fuentes_de_datos.md).

---

## Estructura del repositorio

siniestralidad-vial-sv/
├── data/
│   ├── raw/          # datos de entrada (se versionan en este proyecto)
│   │   ├── siniestros.csv
│   │   ├── victimas.csv
│   │   ├── TAB_POB_AREA_1.xlsx        # censo 2024 (crudo oficial)
│   │   ├── Estaciones SLV.xlsx        # catálogo de estaciones
│   │   ├── feriados_elsalvador_2022_2026.xlsx
│   │   └── clima/                     # 6 estaciones Meteostat horarias
│   └── processed/    # datos derivados (se generan en NB03)
├── notebooks/
│   ├── NB01_descripcion_dataset.ipynb
│   ├── NB02_eda.ipynb
│   ├── NB03_preprocesamiento_features.ipynb
│   └── NB04_modelos_baseline.ipynb
├── src/
│   ├── init.py
│   └── config.py     # rutas, seed=42 y cargadores de datos
├── models/           # modelos entrenados .pkl (se versionan)
├── reports/
│   ├── figures/      # gráficos del EDA
│   └── results/      # métricas .json y tablas comparativas
├── docs/
│   ├── fuentes_de_datos.md    # procedencia, licencias, limitaciones
│   └── diccionario_datos.md   # descripción de cada columna
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

---

## Fuentes de datos

| Fuente              | Contenido                               | Naturaleza                                |
| ------------------- | --------------------------------------- | ----------------------------------------- |
| ONASEVI / FONAT     | Siniestros y víctimas                   | Sintético calibrado a agregados oficiales |
| BCR / ONEC          | Censo de población 2024 (262 distritos) | Real — oficial                            |
| Meteostat           | Clima horario (6 estaciones, 2022–2026) | Real                                      |
| Calendario nacional | Feriados 2022–2026                      | Real                                      |

Detalle, licencias y limitaciones en [`docs/fuentes_de_datos.md`](docs/fuentes_de_datos.md).
Descripción de columnas en [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

---

## Instalación

Requiere **Python 3.12**.

```bash
# 1. Clonar
git clone https://github.com/jpurquilla/siniestralidad-vial-sv.git
cd siniestralidad-vial-sv

# 2. Entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota (macOS):** las dependencias geoespaciales (`geopandas`, `osmnx`)
> requieren GDAL nativo. `geopandas` está fijado en `0.14.4` porque `osmnx`
> exige `geopandas<0.15`. Si `pip` falla en esos paquetes, instálalos con
> `conda install -c conda-forge geopandas osmnx` y el resto con pip. Para la
> Etapa 1 los componentes geoespaciales no son necesarios.

---

## Reproducción

Todo el estado compartido (rutas, semilla, carga de datos) vive en
`src/config.py`. Los notebooks lo importan y **no redefinen** rutas ni semilla:

```python
from src import config as cfg

siniestros = cfg.load_siniestros()
censo = cfg.load_censo()
```

Verificar que el entorno y los datos están correctos:

```bash
python -m src.config
```

Ejecutar los notebooks **en orden** (`NB01 → NB04`). Cada uno consume las
salidas del anterior; NB03 genera los datos derivados en `data/processed/`.

---

## Notebooks (Etapa 1)

| Notebook                               | Contenido                                                          | Criterio de rúbrica                             |
| -------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| **NB01** — Descripción del dataset     | Fuentes, tamaño, variables, calibración                            | Calidad y descripción de datos                  |
| **NB02** — EDA                         | Análisis exploratorio + visualizaciones obligatorias               | EDA completo y visualizaciones                  |
| **NB03** — Preprocesamiento y features | Grilla distrito×fecha×franja, joins (población, clima, calendario) | Preprocesamiento y feature engineering          |
| **NB04** — Modelos baseline            | Predictor ingenuo, Poisson/GLM, Random Forest + SHAP               | Modelos baseline y métricas · Interpretabilidad |

Cada notebook escribe **primero** las celdas de explicación (metodología y
justificación) y luego el código, para que sea legible como documento.

---

## Convenciones del proyecto

- **Reproducibilidad:** `SEED = 42` (definida una sola vez en `config.py`).
- **Configuración centralizada:** rutas y cargadores solo en `src/config.py`.
- **Separación de responsabilidades:** `config.py` hace carga y normalización de
  formato (plomería); las decisiones analíticas viven en los notebooks.
- **Salidas:** figuras en `reports/figures/`, métricas en `reports/results/`
  (`.json`), modelos en `models/` (`.pkl`).
- **Commits:** mensajes en inglés, imperativo (`feat`, `fix`, `docs`, `refactor`…).

---

## Licencia

El **código** de este repositorio se publica con fines académicos. Los **datos**
conservan las licencias de sus fuentes originales (ver
[`docs/fuentes_de_datos.md`](docs/fuentes_de_datos.md)): los datos de clima de
Meteostat bajo CC BY / CC BY-NC 4.0, y el censo como dato oficial público de
BCR/ONEC. Los datos de siniestralidad son sintéticos calibrados a agregados
públicos.