"""Caso de uso: composición de causas (factor_causante) por franja horaria.

Cuenta, sobre el periodo de entrenamiento (2022-2024), qué proporción de los siniestros de
cada franja corresponde a cada causa. Se calcula UNA vez al arrancar y se guarda en memoria.

Python puro: lee los siniestros a través del puerto del dominio.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date

from app.domain.models import ComposicionCausas, FiltroSiniestros, Franja, ProporcionCausa
from app.domain.ports import SiniestroPuntoRepositoryPort

PERIODO_ENTRENAMIENTO_INICIO = date(2022, 1, 1)
PERIODO_ENTRENAMIENTO_FIN = date(2024, 12, 31)


class ComposicionCausasError(RuntimeError):
    """No se pudo calcular una composición válida."""


class CalcularComposicionCausas:
    def __init__(self, siniestros: SiniestroPuntoRepositoryPort) -> None:
        self._siniestros = siniestros

    def ejecutar(self) -> ComposicionCausas:
        # Solo el periodo de entrenamiento: los años posteriores (2025-2026) quedan fuera.
        filtro = FiltroSiniestros(
            fecha_inicio=PERIODO_ENTRENAMIENTO_INICIO, fecha_fin=PERIODO_ENTRENAMIENTO_FIN
        )
        conteos: dict[Franja, Counter[str]] = {f: Counter() for f in Franja}
        for p in self._siniestros.buscar(filtro):
            conteos[p.franja][p.factor_causante] += 1

        por_franja: dict[Franja, tuple[ProporcionCausa, ...]] = {}
        n_siniestros: dict[Franja, int] = {}
        for franja, contador in conteos.items():
            total = sum(contador.values())
            if total == 0:
                raise ComposicionCausasError(
                    f"No hay siniestros de {franja.value} en "
                    f"{PERIODO_ENTRENAMIENTO_INICIO}..{PERIODO_ENTRENAMIENTO_FIN}."
                )
            ordenadas = sorted(contador.items(), key=lambda kv: (-kv[1], kv[0]))
            proporciones = tuple(ProporcionCausa(causa, n / total) for causa, n in ordenadas)
            if not math.isclose(sum(p.proporcion for p in proporciones), 1.0, abs_tol=1e-9):
                raise ComposicionCausasError(f"Las proporciones de {franja.value} no suman 1.")
            por_franja[franja] = proporciones
            n_siniestros[franja] = total

        return ComposicionCausas(
            periodo_inicio=PERIODO_ENTRENAMIENTO_INICIO,
            periodo_fin=PERIODO_ENTRENAMIENTO_FIN,
            n_siniestros=n_siniestros,
            por_franja=por_franja,
        )
