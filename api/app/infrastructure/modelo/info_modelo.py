"""Metadatos del modelo para GET /api/v1/modelo/info: métricas de NB07,
fecha de entrenamiento, variables y limitaciones declaradas. Se lee una sola
vez al arrancar, de los mismos artefactos versionados (nb07_metricas.json,
nb07_especificacion.json) — nunca se hardcodean números aquí."""

from __future__ import annotations

import json
from pathlib import Path

from app.infrastructure.modelo.xgboost_adapter import COLUMNAS_MODELO

_NOMBRE_MODELO_METRICAS = "NB07 - XGBoost (exposición offset)"

# Limitaciones declaradas en NB07_RESUMEN.md / NB09_RESUMEN.md / NB10_RESUMEN.md
# — texto descriptivo, no derivable de un JSON, por eso vive aquí en vez de
# leerse de un archivo.
_LIMITACIONES = [
    "El tercil medio de distritos ordena mal (Spearman ~0.65 en la evaluación "
    "de NB07): el modelo no distingue bien el riesgo relativo dentro de ese grupo.",
    "El conjunto de datos es sintético, calibrado sobre agregados de "
    "ONASEVI/FONAT (NB01), no sobre registros individuales reales.",
    "La calibración (razón predicho/observado) tiene una desviación esperada "
    "del 2-3% por el anclaje del factor de crecimiento en enero de 2022 (NB05).",
    "Los distritos con población menor a 10,000 habitantes (flag poblacion_baja) "
    "producen tasas per cápita inestables; el modelo los señala pero no los filtra.",
    "El submodelo de severidad (NB09) no se expone en esta API: sus variables "
    "(grupo_vulnerable, rango_etario) se conocen después de identificar a las "
    "víctimas, por lo que no admiten uso predictivo ex-ante.",
]


def cargar_info_modelo(ruta_metricas: Path, ruta_especificacion: Path) -> dict:
    metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
    especificacion = json.loads(ruta_especificacion.read_text(encoding="utf-8"))

    return {
        "nombre": "frecuencia_xgboost_offset_v2 (NB07)",
        "fecha_entrenamiento": especificacion.get("generado", "desconocida"),
        "variables": COLUMNAS_MODELO,
        "offset": "log_exposicion_v2 + log_tasa_base (ver NB07 §6.4)",
        "metricas": metricas["modelos"].get(_NOMBRE_MODELO_METRICAS, {}),
        "limitaciones": _LIMITACIONES,
    }
