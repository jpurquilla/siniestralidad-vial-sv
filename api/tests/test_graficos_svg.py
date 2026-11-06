"""Gráficos SVG del informe: bien formados (xml.etree) y con un color fijo por causa."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from app.domain.colores_causas import COLORES_CAUSAS, OTRAS_CAUSAS, colores_para, nombre_causa
from app.domain.informe import CausasDeFranja, FilaCausa, FilaDiaSemana, FilaDistritoFranja
from app.domain.models import Franja, ProporcionCausa
from app.infrastructure.informes import graficos_svg

_SVG = "{http://www.w3.org/2000/svg}"
_CAUSAS = list(COLORES_CAUSAS)[:8]
_COLORES = colores_para(_CAUSAS)


def _por_causa() -> list[FilaCausa]:
    filas = [FilaCausa(c, 100.0 - 10 * i, 80 - 10 * i, 120 - 10 * i, 10.0) for i, c in enumerate(_CAUSAS)]
    return [*filas, FilaCausa(OTRAS_CAUSAS, 12.0, 6, 19, 1.0)]


def _composicion() -> list[CausasDeFranja]:
    todas = [*_CAUSAS, OTRAS_CAUSAS]
    return [CausasDeFranja(f, tuple(ProporcionCausa(c, 1 / 9) for c in todas)) for f in Franja]


def _rect_por_causa(svg: ET.Element) -> list[tuple[str, str]]:
    """(causa, fill) de cada rect cuyo <title> menciona una causa."""
    salida = []
    for rect in svg.iter(f"{_SVG}rect"):
        titulo = rect.find(f"{_SVG}title")
        if titulo is None:
            continue
        for causa in _COLORES:
            if causa in (titulo.text or ""):
                salida.append((causa, rect.get("fill")))
                break
    return salida


@pytest.fixture(params=["causas", "composicion", "dias", "calor"])
def svg(request) -> str:
    if request.param == "causas":
        return graficos_svg.barras_causas_con_rango(_por_causa(), _COLORES)
    if request.param == "composicion":
        return graficos_svg.barras_composicion_por_franja(_composicion(), _COLORES)
    if request.param == "dias":
        return graficos_svg.barras_dia_semana(
            [FilaDiaSemana(d, 10.0 + i, 4, 4 * (10.0 + i)) for i, d in enumerate(("Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"))]
        )
    return graficos_svg.mapa_calor_distrito_franja(
        [FilaDistritoFranja(f"d{i}", f"Distrito & <{i}>", {f: float(i + j) for j, f in enumerate(Franja)}) for i in range(5)]
    )


def test_bien_formado_con_texto_alternativo(svg):
    raiz = ET.fromstring(svg)
    assert raiz.tag == f"{_SVG}svg"
    assert raiz.get("role") == "img"
    assert raiz.get("viewBox")
    assert (raiz.find(f"{_SVG}title").text or "").strip()
    assert (raiz.find(f"{_SVG}desc").text or "").strip()


def test_sin_degradados_ni_imagenes_raster(svg):
    assert "<image" not in svg and "data:image" not in svg


def test_color_fijo_por_causa_en_los_dos_graficos_de_causas():
    graficos = [
        graficos_svg.barras_causas_con_rango(_por_causa(), _COLORES),
        graficos_svg.barras_composicion_por_franja(_composicion(), _COLORES),
    ]
    vistos: dict[str, set[str]] = {}
    for g in graficos:
        pares = _rect_por_causa(ET.fromstring(g))
        assert pares, "el gráfico no tiene barras de causas"
        for causa, fill in pares:
            assert fill == _COLORES[causa]
            vistos.setdefault(causa, set()).add(fill)
    assert set(vistos) == set(_COLORES) & {*_CAUSAS, OTRAS_CAUSAS}
    assert all(len(v) == 1 for v in vistos.values())


def _textos(svg: str) -> list[str]:
    return [t.text or "" for t in ET.fromstring(svg).iter(f"{_SVG}text")]


def test_clave_compacta_pegada_al_grafico_usa_los_mismos_colores():
    svg = graficos_svg.barras_composicion_por_franja(_composicion(), _COLORES)
    rellenos = {r.get("fill") for r in ET.fromstring(svg).iter(f"{_SVG}rect")}
    assert {_COLORES[c] for c in [*_CAUSAS, OTRAS_CAUSAS]} <= rellenos
    assert all(c in _textos(svg) for c in [*_CAUSAS, OTRAS_CAUSAS])  # la clave vive dentro del propio SVG


def test_porcentaje_dentro_del_segmento_solo_si_mide_mas_de_8_por_ciento():
    comp = [CausasDeFranja(Franja.TARDE, (
        ProporcionCausa(_CAUSAS[0], 0.50), ProporcionCausa(_CAUSAS[1], 0.30), ProporcionCausa(_CAUSAS[2], 0.0801),
        ProporcionCausa(_CAUSAS[3], 0.0799), ProporcionCausa(_CAUSAS[4], 0.04), ProporcionCausa(OTRAS_CAUSAS, 0.0),
    ))]
    textos = _textos(graficos_svg.barras_composicion_por_franja(comp, _COLORES))
    assert {"50%", "30%", "8%"} <= set(textos)  # 8.01% sí; 7.99% y 4% no
    assert textos.count("8%") == 1 and "4%" not in textos


@pytest.mark.parametrize(
    ("maximo", "tope", "paso"),
    [(253, 300, 100), (506, 600, 200), (67.4, 80, 20), (100, 100, 20), (9.3, 10, 2), (0.9, 1, 0.2), (1982, 2000, 500)],
)
def test_escala_con_numeros_redondos(maximo, tope, paso):
    t, p = graficos_svg.escala_agradable(maximo)
    assert (t, p) == (tope, paso)
    assert t >= maximo


@pytest.mark.parametrize("maximo", [3, 17.2, 99, 253, 506, 1234, 87654])
def test_pasos_de_la_escala_son_1_2_2_5_o_5_por_potencia_de_10(maximo):
    tope, paso = graficos_svg.escala_agradable(maximo)
    mantisa = paso / 10 ** math.floor(math.log10(paso))
    assert round(mantisa, 6) in (1, 2, 2.5, 5)
    assert tope >= maximo and abs(tope / paso - round(tope / paso)) < 1e-9


def test_los_ejes_de_los_graficos_usan_marcas_redondas():
    etiquetas = [t for t in _textos(graficos_svg.barras_causas_con_rango(_por_causa(), _COLORES)) if t.isdigit()]
    assert "0" in etiquetas and "253" not in etiquetas
    dias = _textos(graficos_svg.barras_dia_semana(
        [FilaDiaSemana(d, 63.0 + i, 4, 0.0) for i, d in enumerate(("L", "M", "X", "J", "V", "S", "D"))]))
    assert {"0", "20", "40", "60", "80"} <= set(dias)


def test_etiqueta_directa_al_final_de_la_barra_con_valor_y_rango():
    textos = _textos(graficos_svg.barras_causas_con_rango(_por_causa(), _COLORES))
    assert "100" in textos and "(80–120)" in textos
    assert not any("Barra:" in t for t in textos)  # sin nota repetida dentro del gráfico


def test_sin_mayusculas_pequenas_ni_letras_espaciadas():
    for g in (graficos_svg.barras_causas_con_rango(_por_causa(), _COLORES),
              graficos_svg.barras_composicion_por_franja(_composicion(), _COLORES)):
        assert "text-transform" not in g and "letter-spacing" not in g


def test_texto_sobre_color_con_contraste_suficiente():
    for color in COLORES_CAUSAS.values():
        texto = graficos_svg.color_texto_sobre(color)
        lum, lum_t = graficos_svg._luminancia(color), graficos_svg._luminancia(texto)
        assert (max(lum, lum_t) + 0.05) / (min(lum, lum_t) + 0.05) >= 3.0, color
    assert graficos_svg.color_texto_sobre("#12355B") == "#ffffff"
    assert graficos_svg.color_texto_sobre("#A9C1D9") != "#ffffff"


def test_paleta_sobria_azules_grises_y_un_solo_acento():
    assert len(COLORES_CAUSAS) == 17
    assert COLORES_CAUSAS["Distracción del conductor"] == "#12355B"
    assert all(k == nombre_causa(k) for k in COLORES_CAUSAS)  # sentence case
    assert COLORES_CAUSAS["Estado de ebriedad o droga"] == "#E3A21A"
    assert [c for c in COLORES_CAUSAS.values() if _es_calido(c)] == ["#E3A21A"]  # ámbar solo para ebriedad
    azules = [COLORES_CAUSAS[c] for c in ("Distracción del conductor", "Invadir carril", "No respetar señales de tránsito")]
    assert len(set(azules)) == 3 and all(_es_azul(c) for c in azules)
    luminancias = [graficos_svg._luminancia(c) for c in azules]
    assert luminancias == sorted(luminancias)  # oscuro, medio, claro
    grises = [COLORES_CAUSAS[c] for c in ("No guardar distancia reglamentaria", "Velocidad excesiva",
                                          "Circular en reversa", "Adelantamiento antireglamentario")]
    assert len({round(graficos_svg._luminancia(c), 2) for c in grises}) == 4  # distinguibles por luminosidad
    assert all(_es_gris(c) for c in grises)
    otras = _COLORES[OTRAS_CAUSAS]
    assert _es_gris(otras) and graficos_svg._luminancia(otras) > max(graficos_svg._luminancia(c) for c in grises)
    assert otras not in COLORES_CAUSAS.values()


def _rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _es_calido(c: str) -> bool:
    r, _, b = _rgb(c)
    return r - b > 60


def _es_azul(c: str) -> bool:
    r, _, b = _rgb(c)
    return b - r > 30


def _es_gris(c: str) -> bool:
    r, g, b = _rgb(c)
    return max(r, g, b) - min(r, g, b) <= 24


def test_nombres_de_causa_en_sentence_case():
    assert nombre_causa("No Guardar Distancia Reglamentaria") == "No guardar distancia reglamentaria"
    assert nombre_causa("Otras causas") == "Otras causas"


def test_causa_desconocida_recibe_color_de_respaldo_estable():
    a = colores_para(["Causa nueva"])
    assert a["Causa nueva"] == colores_para(["Causa nueva"])["Causa nueva"]
    assert a["Causa nueva"] in {"#3d4651", "#76818d", "#a4acb5", "#59636f"}


def test_escapa_caracteres_especiales_en_los_nombres():
    raiz = ET.fromstring(
        graficos_svg.mapa_calor_distrito_franja(
            [FilaDistritoFranja("x", "A & <B>", {f: 1.0 for f in Franja})]
        )
    )
    assert any("A & <B>" in (t.text or "") for t in raiz.iter(f"{_SVG}text"))


def test_grafico_vacio_no_falla():
    ET.fromstring(graficos_svg.mapa_calor_distrito_franja([]))
    ET.fromstring(graficos_svg.barras_dia_semana([]))
