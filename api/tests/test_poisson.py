"""Cuantiles de Poisson en Python puro (dominio): valores conocidos y, si scipy está
disponible, contraste contra scipy.stats.poisson.ppf."""

from __future__ import annotations

import pytest

from app.domain import poisson

# (lambda, cuantil 5%, cuantil 95%) — verificados con scipy.stats.poisson.ppf
_CONOCIDOS = [
    (0.01, 0, 0),
    (0.5, 0, 2),
    (1, 0, 3),
    (3.7, 1, 7),
    (10, 5, 15),
    (50, 39, 62),
    (100, 84, 117),
    (1000, 948, 1052),
    (7000, 6863, 7138),
    (50000, 49632, 50368),
]


@pytest.mark.parametrize(("lam", "inf", "sup"), _CONOCIDOS)
def test_valores_conocidos(lam, inf, sup):
    assert poisson.intervalo_90(lam) == (inf, sup)


def test_lambda_cero_da_cero():
    assert poisson.intervalo_90(0.0) == (0, 0)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_p_fuera_de_rango_falla(p):
    with pytest.raises(ValueError):
        poisson.cuantil(5.0, p)


@pytest.mark.parametrize("lam", [-1.0, float("inf"), float("nan")])
def test_lambda_invalido_falla(lam):
    with pytest.raises(ValueError):
        poisson.cuantil(lam, 0.5)


def test_intervalo_contiene_a_lambda_y_es_creciente():
    anterior = (0, 0)
    for lam in (0.2, 1, 5, 20, 80, 300, 2500):
        inf, sup = poisson.intervalo_90(lam)
        assert inf <= lam <= sup
        assert inf >= anterior[0] and sup >= anterior[1]
        anterior = (inf, sup)


def test_coincide_con_scipy_en_una_malla():
    stats = pytest.importorskip("scipy.stats")
    for lam in [0.003, 0.07, 0.9, 2.2, 6.5, 13.0, 29.9, 77.7, 210.5, 999.9, 4321.0, 25_000.0, 120_000.0]:
        for p in (0.05, 0.5, 0.95):
            assert poisson.cuantil(lam, p) == int(stats.poisson.ppf(p, lam)), (lam, p)
