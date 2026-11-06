"""Contrato de GET /api/v1/modelo/info: metadatos del modelo final (NB12),
sus métricas, pruebas de significancia, importancia SHAP y limitaciones.

Se construye una sola vez al arrancar. Todo número sale de
nb12_metricas.json y nb12_especificacion.json (a través del adaptador del
modelo para el nombre del archivo y la fecha); el único texto fijo son las
limitaciones, que son redacción y no derivan de un JSON — pero las cifras
que citan se verifican contra los JSON al arrancar (ver
`_verificar_cifras_de_limitaciones`), para que no puedan quedar desfasadas.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.models import Franja

_CLAVE_MODELO_FINAL = "ampliado (NB12)"
_CLAVE_REFERENCIA_NB07 = "NB07"
_GRUPO_RURAL = "prueba 2025-2026 | rural (pct_urbano < 0.3)"

LIMITACIONES = [
    "El conjunto de datos es sintético, calibrado sobre agregados de ONASEVI/FONAT; "
    "las reglas con que se generó no están documentadas.",
    "Las variables territoriales (hogares con automóvil, movilidad laboral, "
    "crecimiento del parque) son asociaciones dentro del conjunto, no efectos "
    "causales verificados.",
    "En el periodo de evaluación (2025–2026) el modelo sobreestima un 20% los "
    "siniestros en los distritos rurales (menos de 30% de población urbana) y un "
    "40% en la madrugada (00:00 a 05:59).",
    "El ordenamiento de los distritos por riesgo es algo menos preciso que el del "
    "modelo anterior (correlación de Spearman de 0.945 frente a 0.971).",
    "Para fechas fuera del periodo de enero de 2022 a junio de 2026, el crecimiento "
    "del parque vehicular usa el último mes disponible.",
    "El submodelo de severidad no se expone: sus variables se conocen después de "
    "identificar a las víctimas.",
]


class InfoModeloError(RuntimeError):
    """nb12_metricas.json no tiene la forma esperada o contradice las
    cifras citadas en las limitaciones."""


def _verificar_cifras_de_limitaciones(
    rural: float, madrugada: float, spearman_final: float, spearman_nb07: float
) -> None:
    """Las limitaciones 3 y 4 citan cifras redondeadas; si los JSON cambian y
    ya no coinciden, el arranque falla en vez de servir un texto falso."""
    esperadas = {
        "sobreestima 20% en rural": (round((rural - 1) * 100), 20),
        "sobreestima 40% en madrugada": (round((madrugada - 1) * 100), 40),
        "Spearman global final 0.945": (round(spearman_final, 3), 0.945),
        "Spearman global NB07 0.971": (round(spearman_nb07, 3), 0.971),
    }
    for nombre, (medido, citado) in esperadas.items():
        if medido != citado:
            raise InfoModeloError(
                f"Las limitaciones citan '{nombre}' pero nb12_metricas.json da {medido}."
            )


def cargar_info_modelo(
    ruta_metricas: Path, version_modelo: str, fecha_entrenamiento: str, variables: list[str]
) -> dict:
    try:
        metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
        prueba = metricas["prueba"]
        comparacion = prueba["comparacion"]
        final = comparacion[_CLAVE_MODELO_FINAL]
        referencia = comparacion[_CLAVE_REFERENCIA_NB07]
        mensual = prueba["prueba_mensual"]
        dm = prueba["diebold_mariano"]
        franjas = prueba["calibracion_por_franja"]["ampliado"]
        shap = metricas["importancia_shap"]
        rural = float(metricas["equidad"][_GRUPO_RURAL]["ampliado"])

        calibracion_por_franja = [
            {"franja": franja.value, "calibracion": float(franjas[franja.value])} for franja in Franja
        ]
        importancia_shap = [
            {"variable": variable, "valor": float(valor)}
            for variable, valor in sorted(shap.items(), key=lambda kv: kv[1], reverse=True)
        ]
        pruebas = {
            "t_pareada_mensual": {
                "n_meses": int(mensual["n_meses"]),
                "t": float(mensual["t"]),
                "p": float(mensual["p_t"]),
                "meses_que_mejora": int(mensual["meses_que_mejora"]),
            },
            "diebold_mariano": {
                "DM": float(dm["DM"]),
                "p": float(dm["p"]),
                "dias": int(dm["dias"]),
            },
        }
        _verificar_cifras_de_limitaciones(
            rural=rural,
            madrugada=float(franjas[Franja.MADRUGADA.value]),
            spearman_final=float(final["spearman_global"]),
            spearman_nb07=float(referencia["spearman_global"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise InfoModeloError(f"{ruta_metricas} no tiene la forma esperada: {exc!r}") from exc

    return {
        "nombre": "Modelo de frecuencia de siniestros viales (XGBoost)",
        "version_modelo": version_modelo,
        "fecha_entrenamiento": fecha_entrenamiento,
        "variables": list(variables),
        "offset": "log_exposicion_v2 + log_tasa_base",
        "metricas": dict(final),
        "metricas_referencia_nb07": dict(referencia),
        "pruebas_significancia": pruebas,
        "importancia_shap": importancia_shap,
        "calibracion_por_franja": calibracion_por_franja,
        "limitaciones": list(LIMITACIONES),
    }
