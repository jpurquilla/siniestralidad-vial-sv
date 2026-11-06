"""Nombres y colores de las causas (factor_causante): fuente única para el informe.

Nombres: se presentan en «sentence case» (solo la primera letra en mayúscula) mediante UNA
función, `nombre_causa`; el CSV los trae como «Distracción del Conductor».

Colores: un color fijo por causa, igual en el JSON, el PDF, el Excel y la web (las 8
principales cambian según el periodo, por eso el color va fijado a la causa y no a su
posición en el ranking). Paleta sobria: tres azules del azul institucional (#12355B) para las
tres causas más frecuentes, UN solo acento ámbar para la ebriedad (la causa que varía por
franja), grises de distinta luminosidad para el resto y gris muy claro para «Otras causas».
"""

from __future__ import annotations

OTRAS_CAUSAS = "Otras causas"
NUM_CAUSAS_PRINCIPALES = 8

COLOR_OTRAS_CAUSAS = "#dfe3e8"


def nombre_causa(causa: str) -> str:
    """«Distracción del Conductor» -> «Distracción del conductor» (minúscula salvo la inicial)."""
    texto = " ".join(causa.split())
    return texto[:1].upper() + texto[1:].lower()


EBRIEDAD = nombre_causa("Estado de Ebriedad o Droga")

COLORES_CAUSAS: dict[str, str] = {
    # Las tres más frecuentes: azules (oscuro, medio, claro).
    nombre_causa("Distracción del Conductor"): "#12355B",
    nombre_causa("Invadir Carril"): "#3A6A9B",
    nombre_causa("No Respetar Señales de Tránsito"): "#A9C1D9",
    # Acento único: la causa cuya proporción varía por franja.
    EBRIEDAD: "#E3A21A",
    # Grises (cuatro luminosidades) para las demás, de más a menos frecuente.
    nombre_causa("No Guardar Distancia Reglamentaria"): "#3d4651",
    nombre_causa("Velocidad Excesiva"): "#76818d",
    nombre_causa("Circular en Reversa"): "#a4acb5",
    nombre_causa("Adelantamiento Antireglamentario"): "#59636f",
    nombre_causa("Otros"): "#8c95a0",
    nombre_causa("Inexperiencia"): "#b9c0c8",
    nombre_causa("Giro Incorrecto"): "#4b5560",
    nombre_causa("Imprudencia del Peatón"): "#818b96",
    nombre_causa("Falla Mecánica"): "#aeb5bd",
    nombre_causa("Mal Estado del Vehículo"): "#656f7a",
    nombre_causa("Enfermedad"): "#939ca7",
    nombre_causa("Carga Mal Acondicionada"): "#c2c8cf",
    nombre_causa("Deslumbramiento"): "#707a85",
}

_GRISES_RESPALDO = ("#3d4651", "#76818d", "#a4acb5", "#59636f")


def colores_para(causas: list[str]) -> dict[str, str]:
    """Mapa causa -> color: las conocidas, las de `causas` (gris de respaldo si faltan) y «Otras causas»."""
    colores = dict(COLORES_CAUSAS)
    nuevas = sorted(c for c in causas if c not in colores and c != OTRAS_CAUSAS)
    for i, causa in enumerate(nuevas):
        colores[causa] = _GRISES_RESPALDO[i % len(_GRISES_RESPALDO)]
    colores[OTRAS_CAUSAS] = COLOR_OTRAS_CAUSAS
    return colores
