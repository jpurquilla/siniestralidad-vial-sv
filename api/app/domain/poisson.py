"""Cuantiles de la distribución de Poisson en Python puro (sin scipy).

Se usan para el rango del 90% de los siniestros esperados de un periodo: dado
λ (la suma de los esperados), el rango es [cuantil 5%, cuantil 95%] de
Poisson(λ). Refleja solo la variación aleatoria dado λ.
"""

from __future__ import annotations

import math

# Por debajo de λ − _SIGMAS·√λ − _MARGEN la masa acumulada es despreciable (< 1e-30).
_SIGMAS = 12.0
_MARGEN = 10


def cuantil(lam: float, p: float) -> int:
    """Menor entero k tal que P(X <= k) >= p, con X ~ Poisson(lam)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p debe estar en (0, 1); recibido {p!r}.")
    if lam < 0 or not math.isfinite(lam):
        raise ValueError(f"lam debe ser finito y >= 0; recibido {lam!r}.")
    if lam == 0:
        return 0

    # Se acumula desde un k inicial cercano a la cola izquierda para no recorrer
    # miles de términos sin masa; la pmf se calcula en logaritmos (estable para λ grande)
    # y luego por recurrencia pmf(k+1) = pmf(k)·λ/(k+1).
    k = max(0, int(lam - _SIGMAS * math.sqrt(lam) - _MARGEN))
    pmf = math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
    acumulada = pmf
    # Tolerancia para que p = 0.95 no se pierda por error de punto flotante en la suma.
    objetivo = p - 1e-12
    while acumulada < objetivo:
        k += 1
        pmf *= lam / k
        acumulada += pmf
        if pmf == 0.0:  # cola agotada por underflow: no hay más masa que sumar
            break
    return k


def intervalo_90(lam: float) -> tuple[int, int]:
    """(cuantil 5%, cuantil 95%) de Poisson(lam)."""
    return cuantil(lam, 0.05), cuantil(lam, 0.95)
