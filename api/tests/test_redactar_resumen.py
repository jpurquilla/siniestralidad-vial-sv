"""Redacción del resumen del informe con valores controlados (reglas B.9 del contrato v2)."""

from __future__ import annotations

from datetime import date

import pytest

from app.application.redactar_resumen import conteo, redactar_resumen
from app.domain.informe import DIAS_SEMANA, FilaDiaSemana
from app.domain.models import Franja

_TODAS = list(Franja)
_EBRIEDAD = {Franja.MADRUGADA: 0.16, Franja.MANANA: 0.04, Franja.TARDE: 0.04, Franja.NOCHE: 0.14}


def _dias(promedios: list[float], veces: list[int]) -> list[FilaDiaSemana]:
    return [FilaDiaSemana(DIAS_SEMANA[i], promedios[i] if veces[i] else 0.0, veces[i], promedios[i] * veces[i])
            for i in range(7)]


_VECES_31 = [5, 5, 5, 4, 4, 4, 4]
_PROMEDIOS = [50.0, 51.0, 52.0, 53.0, 60.0, 67.4, 55.0]  # el sábado gana por promedio


def _resumen(**kw):
    base = dict(
        fecha_inicio=date(2026, 11, 2),
        fecha_fin=date(2026, 12, 2),
        total=1982.4,
        ic90=(1909, 2055),
        distritos=[("San Salvador", 225.1), ("San Miguel", 150.0), ("Santa Tecla", 130.0), ("Santa Ana", 110.0)]
        + [(f"Otro {k}", 10.0) for k in range(99)],
        por_franja=[(Franja.MADRUGADA, 300.0), (Franja.MANANA, 500.0), (Franja.TARDE, 680.0), (Franja.NOCHE, 502.4)],
        por_dia=_dias(_PROMEDIOS, _VECES_31),
        ebriedad_por_franja=_EBRIEDAD,
    )
    base.update(kw)
    return redactar_resumen(**base)


def test_forma_completa_del_parrafo_y_las_4_filas():
    r = _resumen()
    # 680 / 1982.4 = 34.3% -> 34; 615.1 / 1982.4 = 31.0% -> 31; noche+madrugada = 15%; día = 4%
    assert r.texto == (
        "Entre el 2 de noviembre y el 2 de diciembre de 2026 se esperan unos 1,982 siniestros en los 103 "
        "distritos (entre 1,909 y 2,055 con 90% de probabilidad). La tarde concentra el 34% y los sábados "
        "son, en promedio, los días de mayor riesgo. San Salvador, San Miguel, Santa Tecla y Santa Ana "
        "reúnen el 31% del total. De noche y de madrugada, la ebriedad explica cerca del 15% de los "
        "siniestros esperados, frente al 4% del día."
    )
    assert [(f.etiqueta, f.valor) for f in r.filas] == [
        ("Total esperado", "1,982 (entre 1,909 y 2,055)"),
        ("Franja de mayor riesgo", "Tarde (12-17), 34% del total"),
        ("Día de mayor riesgo (promedio)", "Sábado, 67 por día"),
        ("Distrito con más siniestros esperados", "San Salvador, 225.1"),
    ]


def test_el_dia_lo_decide_el_promedio_no_las_apariciones():
    # lunes, martes y miércoles aparecen 5 veces (más suma), pero el domingo tiene mayor promedio
    r = _resumen(por_dia=_dias([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 80.0], _VECES_31))
    assert "los domingos son, en promedio" in r.texto
    assert r.filas[2].valor == "Domingo, 80 por día"


@pytest.mark.parametrize(
    ("dia", "plural"),
    [("Lunes", "lunes"), ("Martes", "martes"), ("Miércoles", "miércoles"), ("Jueves", "jueves"),
     ("Viernes", "viernes"), ("Sábado", "sábados"), ("Domingo", "domingos")],
)
def test_concordancia_del_dia(dia, plural):
    promedios = [10.0] * 7
    promedios[DIAS_SEMANA.index(dia)] = 99.0
    assert f"los {plural} son, en promedio, los días de mayor riesgo" in _resumen(por_dia=_dias(promedios, _VECES_31)).texto


def test_empate_de_dias_es_determinista_y_no_favorece_a_los_que_aparecen_mas():
    r = _resumen(por_dia=_dias([40.0] * 7, [4, 4, 4, 5, 5, 5, 4]))
    assert r.filas[2].valor == "Lunes, 40 por día"
    assert r == _resumen(por_dia=_dias([40.0] * 7, [4, 4, 4, 5, 5, 5, 4]))


def test_fechas_en_otros_anios_y_mismo_dia_o_mes():
    assert _resumen(fecha_inicio=date(2026, 12, 20), fecha_fin=date(2027, 1, 5)).texto.startswith(
        "Entre el 20 de diciembre de 2026 y el 5 de enero de 2027 se esperan unos 1,982 siniestros")
    assert _resumen(fecha_inicio=date(2026, 11, 2), fecha_fin=date(2026, 11, 20)).texto.startswith(
        "Entre el 2 y el 20 de noviembre de 2026 se esperan")
    assert _resumen(fecha_inicio=date(2026, 11, 2), fecha_fin=date(2026, 11, 2)).texto.startswith(
        "El 2 de noviembre de 2026 se esperan unos 1,982 siniestros")


def test_un_solo_distrito_singular_y_sin_frase_de_distritos_principales():
    r = _resumen(distritos=[("Santa Ana", 1982.4)])
    assert "en el distrito Santa Ana (entre 1,909" in r.texto
    assert "reúnen" not in r.texto and "reúne" not in r.texto
    assert r.filas[3].valor == "Santa Ana, 1,982.4"


def test_menos_de_cuatro_distritos_usa_los_disponibles():
    r = _resumen(distritos=[("A", 1000.0), ("B", 500.0), ("C", 482.4)])
    assert "en los 3 distritos" in r.texto
    assert "A, B y C reúnen el 100% del total." in r.texto
    r2 = _resumen(distritos=[("A", 1500.0), ("B", 482.4)])
    assert "A y B reúnen el 100% del total." in r2.texto


def test_una_sola_franja_filtrada_omite_la_frase_de_franja():
    r = _resumen(por_franja=[(Franja.NOCHE, 1982.4)], ebriedad_por_franja={Franja.NOCHE: 0.14})
    assert "concentra" not in r.texto
    assert "Los sábados son, en promedio, los días de mayor riesgo." in r.texto
    assert r.texto.endswith("De noche, la ebriedad explica cerca del 14% de los siniestros esperados.")
    assert r.filas[1].valor == "Noche (18-23), 100% del total"


def test_un_solo_dia_de_la_semana_filtrado_omite_la_frase_del_dia():
    solo_martes = _dias([0, 70.0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0])
    r = _resumen(por_dia=solo_martes)
    assert "La tarde concentra el 34% del total." in r.texto
    assert "días de mayor riesgo" not in r.texto
    assert r.filas[2].valor == "Martes, 70 por día"


def test_una_franja_y_un_dia_omite_ambas_frases():
    r = _resumen(por_franja=[(Franja.TARDE, 1982.4)], por_dia=_dias([0, 70.0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0]),
                 ebriedad_por_franja={Franja.TARDE: 0.04})
    assert "concentra" not in r.texto and "mayor riesgo" not in r.texto
    assert "San Salvador, San Miguel, Santa Tecla y Santa Ana reúnen el 31% del total. Por la tarde, la ebriedad" in r.texto


@pytest.mark.parametrize(
    ("franjas", "final"),
    [
        ([Franja.MANANA, Franja.TARDE], "De día, la ebriedad explica cerca del 4% de los siniestros esperados."),
        ([Franja.MANANA], "Por la mañana, la ebriedad explica cerca del 4% de los siniestros esperados."),
        ([Franja.MADRUGADA, Franja.NOCHE],
         "De noche y de madrugada, la ebriedad explica cerca del 15% de los siniestros esperados."),
        ([Franja.NOCHE, Franja.TARDE],
         "De noche, la ebriedad explica cerca del 14% de los siniestros esperados, frente al 4% de la tarde."),
        ([Franja.MADRUGADA, Franja.MANANA],
         "De madrugada, la ebriedad explica cerca del 16% de los siniestros esperados, frente al 4% de la mañana."),
    ],
)
def test_frase_de_ebriedad_segun_las_franjas_filtradas(franjas, final):
    r = _resumen(por_franja=[(f, 500.0) for f in franjas], ebriedad_por_franja={f: _EBRIEDAD[f] for f in franjas})
    assert r.texto.endswith(final)


def test_sin_ebriedad_en_la_composicion_no_hay_frase():
    assert "ebriedad" not in _resumen(ebriedad_por_franja={}).texto


def test_total_de_uno_usa_singular():
    r = _resumen(total=1.2, ic90=(0, 4), distritos=[("Santa Ana", 1.2)], por_franja=[(Franja.TARDE, 1.2)],
                 por_dia=_dias([1.2] * 7, [1] * 7), fecha_fin=date(2026, 11, 8))
    assert "se espera cerca de 1 siniestro en el distrito Santa Ana (entre 0 y 4" in r.texto


def test_conteo_con_separador_de_miles_y_half_up():
    assert [conteo(v) for v in (1982.4, 1982.5, 0.4, 1234567.0)] == ["1,982", "1,983", "0", "1,234,567"]


def test_sin_viñetas_ni_saltos():
    texto = _resumen().texto
    assert "•" not in texto and "\n" not in texto and "  " not in texto
