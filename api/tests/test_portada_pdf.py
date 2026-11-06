"""Portada del informe PDF: una página, mapa por distrito y rótulos sin superponerse."""

from __future__ import annotations

import io
import re
from datetime import date

import pytest

from app.application.generar_informe_periodo import PeticionInformePeriodo
from app.application.redactar_resumen import frase_distritos_principales, periodo_portada
from app.infrastructure.informes import portada_svg
from app.infrastructure.informes.renderer_pdf import RendererPdf, filtros_portada

_FUERA = portada_svg.FUERA["fill"]
_PERIODO = (date(2026, 11, 2), date(2026, 12, 2))


@pytest.fixture(scope="module")
def renderer() -> RendererPdf:
    return RendererPdf()


def _informe(client, **filtros):
    inicio, fin = filtros.pop("periodo", _PERIODO)
    return client.app.state.generar_informe_periodo_uc.ejecutar(PeticionInformePeriodo(inicio, fin, **filtros))


def _mapa(renderer: RendererPdf, informe) -> str:
    return str(renderer._contexto_portada(informe, 31)["mapa_svg"])


CASOS = {
    "un-distrito": dict(distritos=("san salvador",)),
    "un-departamento": dict(departamento="La Libertad"),
    "los-103": dict(),
}


class TestPortadaUnaPagina:
    @pytest.mark.parametrize("filtros", CASOS.values(), ids=CASOS.keys())
    def test_la_portada_cabe_en_una_pagina_y_el_resumen_empieza_en_la_siguiente(self, client, renderer, filtros):
        pypdf = pytest.importorskip("pypdf")
        informe = _informe(client, **filtros)
        pdf = renderer.renderizar(informe)
        with pypdf.apply_configuration(stream_parsing_work_maximum_count=50_000_000):
            paginas = pypdf.PdfReader(io.BytesIO(pdf)).pages
            primera, segunda = paginas[0].extract_text(), paginas[1].extract_text()
        assert "Siniestros esperados" in primera and "Radar Vial SV" in primera
        assert "siniestros esperados en" in primera and "con 90% de probabilidad" in primera
        assert "Datos sintéticos, calibrados sobre agregados de ONASEVI/FONAT" in primera  # pie de la portada
        assert "Página 1 de" not in primera  # la portada solo lleva el aviso
        assert segunda.lstrip().startswith("Resumen")

    def test_la_portada_trae_los_valores_calculados(self, client, renderer):
        informe = _informe(client)
        c = renderer._contexto_portada(informe, 31)
        assert c["periodo_texto"] == "2 de noviembre al 2 de diciembre de 2026"
        assert c["total"] == f"{round(informe.total.esperado):,d}"
        assert (c["ic_inf"], c["ic_sup"]) == (f"{informe.total.ic90_inf:,d}", f"{informe.total.ic90_sup:,d}")
        assert c["dias"] == "31"
        assert c["filtros"] == [("Distritos", "Los 103 del estudio"), ("Franjas y días", "Todas las franjas y todos los días")]
        assert re.fullmatch(r"\d{1,2} de \w+ de \d{4}, \d{2}:\d{2}", c["generado"])
        assert c["top_texto"].endswith("% del total.") and "reúnen" in c["top_texto"]


class TestMapaDeLaPortada:
    def test_con_un_solo_distrito_lo_colorea_y_deja_el_resto_fuera_del_estudio(self, client, renderer):
        svg = _mapa(renderer, _informe(client, distritos=("san salvador",)))
        rellenos = re.findall(r'<path d="[^"]+" fill="(#[0-9A-Fa-f]{6})"', svg)
        coloreados = [f for f in rellenos if f != _FUERA]
        assert coloreados == [portada_svg.ESCALA[-1]]  # uno solo, con el azul más oscuro
        assert len(rellenos) - 1 == rellenos.count(_FUERA) == 261  # los otros 261 de los 262, fuera del estudio

    def test_con_los_103_colorea_103_y_deja_159_fuera(self, client, renderer):
        svg = _mapa(renderer, _informe(client))
        rellenos = re.findall(r'<path d="[^"]+" fill="(#[0-9A-Fa-f]{6})"', svg)
        assert rellenos.count(_FUERA) == 159 and len(rellenos) - rellenos.count(_FUERA) == 103

    def test_la_leyenda_de_un_solo_distrito_es_una_clase(self, client, renderer):
        valores = {("San Salvador", "San Salvador"): 12.34}
        _, leyenda = portada_svg.mapa_distritos_svg(renderer._geo(), valores)
        assert len(leyenda) == 1 and leyenda[0]["etiqueta"] == "12"

    def test_la_leyenda_completa_tiene_cinco_clases_sin_repetir(self, client, renderer):
        informe = _informe(client)
        valores = {(d.departamento, d.distrito): d.esperado for d in informe.por_distrito}
        _, leyenda = portada_svg.mapa_distritos_svg(renderer._geo(), valores)
        assert [c["color"] for c in leyenda] == portada_svg.ESCALA
        assert len({c["etiqueta"] for c in leyenda}) == 5

    @pytest.mark.parametrize("filtros", CASOS.values(), ids=CASOS.keys())
    def test_los_rotulos_no_se_superponen_ni_se_salen_del_mapa(self, client, renderer, filtros):
        svg = _mapa(renderer, _informe(client, **filtros))
        rotulos = re.findall(r'<text x="([\d.]+)" y="[-\d.]+" class="n">([^<]+)</text><text x="[\d.]+" y="[-\d.]+" class="v">([^<]+)</text>', svg)
        assert rotulos, "no hay rótulos"
        cajas = [(float(x), float(x) + portada_svg.ancho_rotulo(n, v)) for x, n, v in rotulos]
        for (_, fin), (inicio, _) in zip(cajas, cajas[1:], strict=False):
            assert fin <= inicio, cajas  # cada rótulo empieza después de que termina el anterior
        assert cajas[0][0] >= 0 and cajas[-1][1] <= 1000.0 + 1e-6

    def test_posiciones_de_rotulos_resuelve_apreturas_en_el_borde_derecho(self):
        xs = portada_svg.posiciones_rotulos([850, 900, 960, 990], [120, 150, 100, 130], 1000.0)
        cajas = [(x, x + w) for x, w in zip(xs, [120, 150, 100, 130], strict=True)]
        assert all(a[1] + 14 <= b[0] + 1e-9 for a, b in zip(cajas, cajas[1:], strict=False))
        assert cajas[-1][1] <= 1000.0 and xs == sorted(xs)

    def test_los_nombres_de_distritos_se_escapan_en_el_svg(self):
        geo = {"features": [{"properties": {"Dep_min": "X", "Dis_min": "A&B <1>"},
                             "geometry": {"type": "Polygon", "coordinates": [[[-89, 14], [-88.9, 14], [-88.9, 13.9], [-89, 14]]]}}]}
        svg, _ = portada_svg.mapa_distritos_svg(geo, {("X", "A&B <1>"): 5.0})
        assert "A&amp;B &lt;1&gt;" in svg and "A&B <1>" not in svg


class TestTextosDeLaPortada:
    @pytest.mark.parametrize(
        ("inicio", "fin", "esperado"),
        [
            (date(2026, 11, 2), date(2026, 12, 2), "2 de noviembre al 2 de diciembre de 2026"),
            (date(2026, 11, 2), date(2026, 11, 30), "2 al 30 de noviembre de 2026"),
            (date(2026, 11, 2), date(2026, 11, 2), "2 de noviembre de 2026"),
            (date(2026, 12, 30), date(2027, 1, 2), "30 de diciembre de 2026 al 2 de enero de 2027"),
        ],
    )
    def test_periodo_de_la_portada(self, inicio, fin, esperado):
        assert periodo_portada(inicio, fin) == esperado

    def test_frase_de_distritos_principales(self):
        d = [("San Salvador", 50.0), ("San Miguel", 30.0), ("Santa Tecla", 10.0), ("Santa Ana", 5.0), ("Apopa", 5.0)]
        assert frase_distritos_principales(d, 100.0) == "San Salvador, San Miguel, Santa Tecla y Santa Ana reúnen el 95% del total."
        assert frase_distritos_principales(d[:2], 100.0) == "San Salvador y San Miguel reúnen el 80% del total."
        assert frase_distritos_principales(d[:1], 100.0) == ""

    def test_filtros_en_lenguaje_llano(self, client):
        assert filtros_portada(_informe(client, departamento="La Libertad"))[0] == ("Departamento", "La Libertad")
        assert filtros_portada(_informe(client, distritos=("san salvador",)))[0] == ("Distritos", "San Salvador")
        assert filtros_portada(_informe(client))[0] == ("Distritos", "Los 103 del estudio")
        f = filtros_portada(_informe(client, franjas=(__import__("app.domain.models", fromlist=["Franja"]).Franja.NOCHE,), dias_semana=(5, 6)))
        assert f[1] == ("Franjas y días", "Franjas: Noche (18-23). Días: sábado y domingo")
