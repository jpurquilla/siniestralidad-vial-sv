"""Reglas de calendario ex-ante — NB03 §5.1. Python puro: solo dependen de la
fecha, no de ningún archivo ni framework."""

from __future__ import annotations

from datetime import date


def es_lluviosa(fecha: date) -> int:
    """1 si el mes cae en mayo-octubre (temporada lluviosa)."""
    return int(5 <= fecha.month <= 10)


def dia_semana(fecha: date) -> int:
    """0=lunes ... 6=domingo (pandas/python weekday, NB03 §5.1)."""
    return fecha.weekday()


def es_finde(fecha: date) -> int:
    """1 si el día es viernes o sábado — el par de mayor siniestralidad
    según NB02, no el fin de semana calendario (sáb-dom)."""
    return int(fecha.weekday() in (4, 5))


def periodo_agostino(fecha: date) -> int:
    """1 si la fecha cae en la ventana nacional 3-6 de agosto (Fiestas
    Agostinas), aplicada a todos los distritos."""
    return int(fecha.month == 8 and 3 <= fecha.day <= 6)
