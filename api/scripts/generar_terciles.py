"""Genera api/app/infrastructure/modelo/terciles_nb07.json.

Script de UNA sola vez (no corre en el arranque del servicio). Replica
exactamente el protocolo de NB07 §5.4 y nb07_metricas.json ("definicion_tercil_alto":
"los 35 distritos con mayor total observado en el conjunto evaluado"):

  1. Agrega, por distrito, el total de n_siniestros OBSERVADO en el conjunto
     de test (2025-01-01 a 2026-06-30, columna split=='test').
  2. Ordena descendente por ese total observado y parte en tres grupos fijos:
     alto = los 35 distritos con mayor total; medio = los siguientes 34;
     bajo = los últimos 34.
  3. Calcula, dentro de cada grupo, el Spearman entre el total OBSERVADO y el
     total PREDICHO por el modelo NB07 (frecuencia_xgboost_offset_v2.pkl) —
     el mismo indicador de confiabilidad de ordenamiento que reporta
     NB07_RESUMEN.md §5.4.

Solo lee artefactos existentes (matriz_frecuencia_v2.csv, el .pkl,
nb07_especificacion.json) — no modifica nada fuera de api/.

Uso: api/.venv/bin/python api/scripts/generar_terciles.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

_API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_DIR))

from app.config import settings  # noqa: E402
from app.infrastructure.modelo.xgboost_adapter import COLUMNAS_MODELO  # noqa: E402

N_ALTO = 35
N_MEDIO = 34
N_BAJO = 34


def main() -> None:
    especificacion = json.loads(settings.ruta_especificacion_nb07.read_text(encoding="utf-8"))
    info = especificacion["modelos"]["frecuencia_xgboost_offset_v2.pkl"]
    log_tasa_base = float(info["log_tasa_base"])
    assert info["columnas"] == COLUMNAS_MODELO

    booster = joblib.load(settings.ruta_modelo_frecuencia)

    df = pd.read_csv(settings.ruta_matriz_frecuencia)
    test = df[df["split"] == "test"].copy()
    assert len(test) > 0, "no hay filas de test en la matriz"

    matriz = test[COLUMNAS_MODELO].to_numpy(dtype=float)
    margen = (test["log_exposicion_v2"] + log_tasa_base).to_numpy(dtype=float)
    dmatrix = xgb.DMatrix(matriz, base_margin=margen, feature_names=COLUMNAS_MODELO)
    test["predicho"] = booster.predict(dmatrix)

    agregado = test.groupby("distrito_key").agg(
        observado_total=("n_siniestros", "sum"),
        predicho_total=("predicho", "sum"),
    )
    assert len(agregado) == 103, f"se esperaban 103 distritos, hay {len(agregado)}"

    agregado = agregado.sort_values("observado_total", ascending=False)
    assert N_ALTO + N_MEDIO + N_BAJO == len(agregado)

    tercil_por_distrito: dict[str, str] = {}
    for i, distrito_key in enumerate(agregado.index):
        if i < N_ALTO:
            tercil_por_distrito[distrito_key] = "alto"
        elif i < N_ALTO + N_MEDIO:
            tercil_por_distrito[distrito_key] = "medio"
        else:
            tercil_por_distrito[distrito_key] = "bajo"

    agregado["tercil"] = agregado.index.map(tercil_por_distrito)

    confiabilidad = {}
    for tercil, grupo in agregado.groupby("tercil"):
        spearman = grupo["observado_total"].corr(grupo["predicho_total"], method="spearman")
        confiabilidad[tercil] = {"n_distritos": int(len(grupo)), "spearman": round(float(spearman), 6)}

    salida = {
        "fuente": "matriz_frecuencia_v2.csv (split=='test', 2025-01-01 a 2026-06-30) "
        "+ models/frecuencia_xgboost_offset_v2.pkl",
        "protocolo": "NB07 §5.4 / nb07_metricas.json.protocolo.definicion_tercil_alto: "
        "35/34/34 distritos ordenados por total OBSERVADO en el conjunto de test",
        "generado_por": "api/scripts/generar_terciles.py",
        "tercil_por_distrito": tercil_por_distrito,
        "confiabilidad_por_tercil": confiabilidad,
    }

    ruta_salida = _API_DIR / "app" / "infrastructure" / "modelo" / "terciles_nb07.json"
    ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Escrito {ruta_salida}")
    for tercil, info_tercil in confiabilidad.items():
        print(f"  {tercil}: n={info_tercil['n_distritos']} spearman={info_tercil['spearman']}")


if __name__ == "__main__":
    main()
