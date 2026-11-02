"""Genera api/app/infrastructure/modelo/terciles_nb12.json.

Script de UNA sola vez (no corre en el arranque del servicio). Replica
el protocolo de NB07 §5.4 y nb07_metricas.json ("definicion_tercil_alto":
"los 35 distritos con mayor total observado en el conjunto evaluado"), pero
con el MODELO FINAL de NB12:

  1. Agrega, por distrito, el total de n_siniestros OBSERVADO en el conjunto
     de test (2025-01-01 a 2026-06-30, columna split=='test' de
     matriz_frecuencia_v3.csv.gz).
  2. Ordena descendente por ese total observado y parte en tres grupos fijos:
     alto = los 35 distritos con mayor total; medio = los siguientes 34;
     bajo = los últimos 34.
  3. Calcula, dentro de cada grupo, el Spearman entre el total OBSERVADO y el
     total PREDICHO por el modelo final (el que dicta nb12_especificacion.json).

El Spearman por tercil es DESCRIPTIVO: según NB08 depende más del ancho del
tercil que del modelo, así que no debe leerse como una medida de calidad del
modelo.

La predicción pasa por XGBoostFrecuenciaAdapter.predecir_lote(), el único
lugar que suma log_tasa_base al base_margin. No toca terciles_nb07.json.

Uso: api/.venv/bin/python api/scripts/generar_terciles.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_DIR))

from app.config import settings  # noqa: E402
from app.domain.models import COLUMNAS_MODELO, FeaturesCelda  # noqa: E402
from app.infrastructure.modelo.xgboost_adapter import XGBoostFrecuenciaAdapter  # noqa: E402

N_ALTO = 35
N_MEDIO = 34
N_BAJO = 34


def main() -> None:
    adaptador = XGBoostFrecuenciaAdapter(settings.directorio_modelos, settings.ruta_especificacion_nb12)
    adaptador.cargar_y_validar()

    df = pd.read_csv(settings.ruta_matriz_frecuencia)
    test = df[df["split"] == "test"].copy()
    assert len(test) > 0, "no hay filas de test en la matriz"

    filas = [
        FeaturesCelda(
            **{c: getattr(r, c) for c in COLUMNAS_MODELO}, log_exposicion_v2=r.log_exposicion_v2
        )
        for r in test.itertuples(index=False)
    ]
    test["predicho"] = adaptador.predecir_lote(filas)

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
        "fuente": "matriz_frecuencia_v3.csv.gz (split=='test', 2025-01-01 a 2026-06-30) "
        f"+ models/{adaptador.version_modelo}",
        "protocolo": "NB07 §5.4 / nb07_metricas.json.protocolo.definicion_tercil_alto: "
        "35/34/34 distritos ordenados por total OBSERVADO en el conjunto de test",
        "nota": "El Spearman por tercil es descriptivo: segun NB08 depende mas del ancho "
        "del tercil que del modelo.",
        "generado_por": "api/scripts/generar_terciles.py",
        "tercil_por_distrito": tercil_por_distrito,
        "confiabilidad_por_tercil": confiabilidad,
    }

    ruta_salida = _API_DIR / "app" / "infrastructure" / "modelo" / "terciles_nb12.json"
    ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Escrito {ruta_salida}")
    for tercil, info_tercil in confiabilidad.items():
        print(f"  {tercil}: n={info_tercil['n_distritos']} spearman={info_tercil['spearman']}")


if __name__ == "__main__":
    main()
