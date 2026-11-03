"""Textos compartidos por el PDF y el Excel del informe (una sola redacción)."""

from __future__ import annotations

from app.domain.informe import InformePeriodo
from app.infrastructure.informes import formato

MAX_DISTRITOS_MAPA_CALOR = 20

FUENTE_DATOS = (
    "Fuente: Radar Vial SV, modelo de frecuencia (NB12), con datos sintéticos calibrados a ONASEVI/FONAT."
)




def supuestos_texto(i: InformePeriodo) -> list[str]:
    """Supuestos del informe (el mismo texto en el PDF y el Excel)."""
    s = i.supuestos
    textos = ["La lluvia corresponde a la normal climatológica de cada mes."]
    if s.dias_con_ultimo_parque > 0:
        textos.append(
            "En fechas sin dato de parque vehicular se usa el último dato disponible "
            f"({s.dias_con_ultimo_parque} de {s.dias_con_lluvia_normal} días)."
        )
    vehiculos = ", ".join(f"{v.tipo} {formato.porcentaje(v.porcentaje)}" for v in s.composicion_vehiculos)
    textos += [
        "Los valores son promedios (esperados), no conteos observados.",
        "El esperado por causa es el esperado de cada franja multiplicado por la proporción "
        "histórica (2022-2024) de esa causa en la franja.",
        "El esperado por día de la semana es un promedio: la suma de ese día en el periodo dividida "
        "entre las veces que el día aparece (Figura 3 y Tabla 2).",
        "El tercil agrupa a los distritos del informe según su esperado en el periodo (alto, el "
        "tercio superior; medio; bajo, el tercio inferior); la tasa es el esperado por cada 10,000 "
        "habitantes (población total del distrito).",
    ]
    if len(i.distrito_x_franja) > MAX_DISTRITOS_MAPA_CALOR:
        textos.append(
            f"La Figura 4 muestra los {MAX_DISTRITOS_MAPA_CALOR} distritos con más siniestros "
            f"esperados de {len(i.distrito_x_franja)}."
        )
    textos.append(f"Tipo de vehículo en los datos 2022-2024 (no se desglosa): {vehiculos}.")
    return textos
