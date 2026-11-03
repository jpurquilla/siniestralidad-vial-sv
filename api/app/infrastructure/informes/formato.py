"""Formato de números y etiquetas compartido por el PDF y los gráficos (punto decimal,
coma de miles, como en el resto de la API)."""

from __future__ import annotations

from app.domain.models import Franja

# Etiquetas legibles de las franjas (el valor del enum no lleva tilde: «Manana»).
NOMBRE_FRANJA: dict[Franja, str] = {
    Franja.MADRUGADA: "Madrugada",
    Franja.MANANA: "Mañana",
    Franja.TARDE: "Tarde",
    Franja.NOCHE: "Noche",
}
HORARIO_FRANJA: dict[Franja, str] = {
    Franja.MADRUGADA: "00:00–05:59",
    Franja.MANANA: "06:00–11:59",
    Franja.TARDE: "12:00–17:59",
    Franja.NOCHE: "18:00–23:59",
}


def numero(valor: float, decimales: int = 1) -> str:
    return f"{valor:,.{decimales}f}"


def entero(valor: int) -> str:
    return f"{valor:,d}"


def porcentaje(valor: float, decimales: int = 1) -> str:
    return f"{valor:.{decimales}f}%"


def etiqueta_franja(franja: Franja | str) -> str:
    """«Madrugada (00-05)», «Mañana (06-11)»…: el valor del enum con la tilde de presentación."""
    return Franja(franja).value.replace("Manana", "Mañana")
