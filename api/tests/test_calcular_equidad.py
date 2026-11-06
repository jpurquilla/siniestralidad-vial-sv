"""Tests del caso de uso de equidad — lógica pura, con dobles en memoria
para los puertos (no toca ningún archivo real ni el modelo)."""

from __future__ import annotations

import pytest

from app.application.calcular_equidad import CalcularEquidad
from app.domain.models import ComparacionDistrito, Distrito


class ComparacionFalsa:
    def __init__(self, comparaciones: list[ComparacionDistrito]) -> None:
        self._comparaciones = comparaciones

    def por_distrito(self) -> list[ComparacionDistrito]:
        return list(self._comparaciones)


class DistritoRepositorioFalso:
    def __init__(self, distritos: list[Distrito]) -> None:
        self._distritos = distritos

    def listar(self) -> list[Distrito]:
        return list(self._distritos)

    def obtener(self, codigo):
        return next((d for d in self._distritos if d.codigo == codigo), None)

    def existe(self, codigo):
        return any(d.codigo == codigo for d in self._distritos)


def _distrito(codigo: str, pct_urbano: float, poblacion_baja: bool) -> Distrito:
    return Distrito(
        codigo=codigo,
        nombre=codigo,
        departamento="X",
        pct_urbano=pct_urbano,
        poblacion=10_000,
        poblacion_baja=poblacion_baja,
        pct_hogares_auto=0.3,
        pct_trabaja_otro_lugar=0.1,
    )


class TestClasificacionPorUmbral:
    def test_separa_urbano_y_rural_por_el_umbral(self):
        distritos = [
            _distrito("a", pct_urbano=0.9, poblacion_baja=False),
            _distrito("b", pct_urbano=0.1, poblacion_baja=False),
        ]
        comparaciones = [
            ComparacionDistrito("a", observado=10, predicho=10),
            ComparacionDistrito("b", observado=5, predicho=5),
        ]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        resultado = caso_uso.ejecutar()

        urbano = next(g for g in resultado.urbano_rural if g.grupo == "Urbano")
        rural = next(g for g in resultado.urbano_rural if g.grupo == "Rural")
        assert urbano.n_distritos == 1
        assert rural.n_distritos == 1

    def test_el_distrito_justo_en_el_umbral_es_urbano(self):
        distritos = [_distrito("a", pct_urbano=0.3, poblacion_baja=False)]
        comparaciones = [ComparacionDistrito("a", observado=1, predicho=1)]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        resultado = caso_uso.ejecutar()

        assert next(g for g in resultado.urbano_rural if g.grupo == "Urbano").n_distritos == 1


class TestCalibracion:
    def test_calibracion_es_predicho_sobre_observado(self):
        distritos = [
            _distrito("a", pct_urbano=0.9, poblacion_baja=False),
            _distrito("b", pct_urbano=0.9, poblacion_baja=False),
        ]
        comparaciones = [
            ComparacionDistrito("a", observado=100, predicho=80),
            ComparacionDistrito("b", observado=50, predicho=40),
        ]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        resultado = caso_uso.ejecutar()

        urbano = next(g for g in resultado.urbano_rural if g.grupo == "Urbano")
        # (80+40) / (100+50) = 120/150 = 0.8: el modelo subestima un 20%
        assert urbano.calibracion == pytest.approx(0.8)

    def test_sobreestimacion_da_calibracion_mayor_a_uno(self):
        distritos = [_distrito("r", pct_urbano=0.1, poblacion_baja=False)]
        comparaciones = [ComparacionDistrito("r", observado=100, predicho=120)]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        rural = next(g for g in caso_uso.ejecutar().urbano_rural if g.grupo == "Rural")
        assert rural.calibracion == pytest.approx(1.2)


class TestSpearman:
    def test_orden_perfecto_da_spearman_uno(self):
        distritos = [_distrito(c, pct_urbano=0.9, poblacion_baja=False) for c in ("a", "b", "c")]
        comparaciones = [
            ComparacionDistrito("a", observado=10, predicho=1),
            ComparacionDistrito("b", observado=20, predicho=2),
            ComparacionDistrito("c", observado=30, predicho=3),
        ]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        resultado = caso_uso.ejecutar()

        urbano = next(g for g in resultado.urbano_rural if g.grupo == "Urbano")
        assert urbano.spearman == pytest.approx(1.0)

    def test_un_solo_distrito_no_lanza_y_da_spearman_cero(self):
        distritos = [_distrito("a", pct_urbano=0.9, poblacion_baja=False)]
        comparaciones = [ComparacionDistrito("a", observado=10, predicho=8)]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        resultado = caso_uso.ejecutar()

        urbano = next(g for g in resultado.urbano_rural if g.grupo == "Urbano")
        assert urbano.spearman == 0.0


class TestDensidadPoblacional:
    def test_separa_por_flag_poblacion_baja(self):
        distritos = [
            _distrito("a", pct_urbano=0.9, poblacion_baja=False),
            _distrito("b", pct_urbano=0.9, poblacion_baja=True),
        ]
        comparaciones = [
            ComparacionDistrito("a", observado=10, predicho=10),
            ComparacionDistrito("b", observado=1, predicho=1),
        ]
        caso_uso = CalcularEquidad(
            ComparacionFalsa(comparaciones), DistritoRepositorioFalso(distritos), umbral_pct_urbano=0.3
        )
        resultado = caso_uso.ejecutar()

        alta = next(g for g in resultado.densidad_poblacional if g.grupo == "Alta densidad")
        baja = next(g for g in resultado.densidad_poblacional if g.grupo == "Baja densidad")
        assert alta.n_distritos == 1
        assert baja.n_distritos == 1
