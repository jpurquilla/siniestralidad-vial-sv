"""Renderer PDF del informe: plantilla Jinja2 (HTML + CSS) -> WeasyPrint.

WeasyPrint maqueta HTML/CSS con paginación (encabezados y pies repetidos, `break-inside`),
incrusta las fuentes (Public Sans, licencia OFL) y conserva los SVG como vectores. Requiere pango en el sistema
(`brew install pango` / `apt install libpango-1.0-0 libpangoft2-1.0-0`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from app.application.redactar_resumen import fecha_larga, frase_distritos_principales, periodo_portada
from app.config import settings
from app.domain.informe import DIAS_SEMANA, InformePeriodo
from app.infrastructure.informes import formato, graficos_svg, portada_svg
from app.infrastructure.informes.textos import FUENTE_DATOS, MAX_DISTRITOS_MAPA_CALOR, supuestos_texto

_DIRECTORIO = Path(__file__).resolve().parent


def _lista_natural(valores: Sequence[str]) -> str:
    """«A», «A y B», «A, B y C»."""
    if len(valores) <= 1:
        return "".join(valores)
    return ", ".join(valores[:-1]) + " y " + valores[-1]


def filtros_portada(i: InformePeriodo) -> list[tuple[str, str]]:
    """Filtros del informe en lenguaje llano, como pares (etiqueta, valor) para la portada."""
    f = i.parametros.filtros
    nombres = {d.codigo: d.distrito for d in i.por_distrito}
    if f.distritos:
        elegidos = [nombres.get(c, c) for c in f.distritos]
        valor = _lista_natural(elegidos) if len(elegidos) <= 4 else f"{len(elegidos)} distritos elegidos"
        if f.departamento:
            valor += f" ({f.departamento})"
        distritos = ("Distritos", valor)
    elif f.departamento:
        distritos = ("Departamento", f.departamento)
    else:
        n = len(i.por_distrito)
        distritos = ("Distritos", f"Los {n} del estudio" if n == 103 else f"{n} distritos")

    if f.franjas or f.dias_semana:
        partes = []
        if f.franjas:
            partes.append("Franjas: " + _lista_natural([formato.etiqueta_franja(x) for x in f.franjas]))
        if f.dias_semana:
            partes.append("Días: " + _lista_natural([DIAS_SEMANA[d].lower() for d in sorted(f.dias_semana)]))
        franjas_dias = ". ".join(partes)
    else:
        franjas_dias = "Todas las franjas y todos los días"
    return [distritos, ("Franjas y días", franjas_dias)]


def _lista_o_todos(valores: Sequence[str], todos: str) -> str:
    return ", ".join(valores) if valores else todos


class RendererPdf:
    def __init__(self, ruta_geojson_censo: Path | None = None) -> None:
        self._ruta_geojson = ruta_geojson_censo or settings.ruta_geojson_censo
        self._geojson: dict | None = None
        self._logo_svg = (_DIRECTORIO / "logo.svg").read_text(encoding="utf-8")
        self._entorno = Environment(
            loader=FileSystemLoader(_DIRECTORIO),
            autoescape=select_autoescape(["html"]),
        )
        self._entorno.globals.update(
            numero=formato.numero,
            entero=formato.entero,
            porcentaje=formato.porcentaje,
            etiqueta_franja=formato.etiqueta_franja,
        )

    def _geo(self) -> dict:
        """GeoJSON de distritos del censo 2024 (se lee una vez)."""
        if self._geojson is None:
            self._geojson = json.loads(self._ruta_geojson.read_text(encoding="utf-8"))
        return self._geojson

    def _contexto_portada(self, i: InformePeriodo, n_dias: int) -> dict:
        p = i.parametros
        valores = {(d.departamento, d.distrito): d.esperado for d in i.por_distrito}
        mapa_svg, leyenda = portada_svg.mapa_distritos_svg(self._geo(), valores)
        return {
            "periodo_texto": periodo_portada(p.fecha_inicio, p.fecha_fin),
            "dias": formato.entero(n_dias),
            "total": formato.entero(round(i.total.esperado)),
            "ic_inf": formato.entero(i.total.ic90_inf),
            "ic_sup": formato.entero(i.total.ic90_sup),
            "mapa_svg": Markup(mapa_svg),
            "leyenda": leyenda,
            "top_texto": frase_distritos_principales(
                [(d.distrito, d.esperado) for d in i.por_distrito], i.total.esperado
            ),
            "filtros": filtros_portada(i),
            "generado": f"{fecha_larga(i.generado)}, {i.generado:%H:%M}",
            "logo_svg": Markup(self._logo_svg),
        }

    def renderizar_html(self, i: InformePeriodo) -> str:
        """HTML intermedio (con los SVG incrustados): lo que WeasyPrint convierte a PDF."""
        p = i.parametros
        f = p.filtros
        nombres = {d.codigo: d.distrito for d in i.por_distrito}
        n_dias = (p.fecha_fin - p.fecha_inicio).days + 1

        colores = i.colores_causas
        mapa = i.distrito_x_franja[:MAX_DISTRITOS_MAPA_CALOR]
        contexto = {
            "i": i,
            "periodo": f"{fecha_larga(p.fecha_inicio)} al {fecha_larga(p.fecha_fin)}"
            if n_dias > 1 else fecha_larga(p.fecha_inicio),
            "periodo_corto": f"{p.fecha_inicio.isoformat()} a {p.fecha_fin.isoformat()}",
            "n_dias": n_dias,
            "generado": f"{fecha_larga(i.generado)}, {i.generado:%H:%M} (hora de El Salvador)",
            "filtros": {
                "departamento": f.departamento or "Todos",
                "distritos": _lista_o_todos([nombres.get(c, c) for c in f.distritos], "Todos"),
                "franjas": _lista_o_todos([formato.etiqueta_franja(x) for x in f.franjas], "Todas"),
                "dias": _lista_o_todos([DIAS_SEMANA[d] for d in sorted(f.dias_semana)], "Todos"),
            },
            "fuente": FUENTE_DATOS,
            "svg_causas": Markup(graficos_svg.barras_causas_con_rango(i.por_causa, colores)),
            "svg_composicion": Markup(graficos_svg.barras_composicion_por_franja(i.causas_por_franja, colores)),
            "svg_dias": Markup(graficos_svg.barras_dia_semana(i.por_dia_semana)),
            "veces_dias": ", ".join(f"{d.dia.lower()} {d.veces}" for d in i.por_dia_semana),
            "svg_calor": Markup(graficos_svg.mapa_calor_distrito_franja(mapa)),
            "distritos_mapa": len(mapa),
            "supuestos": supuestos_texto(i),
            **self._contexto_portada(i, n_dias),
        }
        return self._entorno.get_template("plantilla_informe.html").render(**contexto)

    def renderizar(self, i: InformePeriodo) -> bytes:
        from weasyprint import CSS, HTML  # import diferido: requiere pango en el sistema
        from weasyprint.text.fonts import FontConfiguration

        html = self.renderizar_html(i)
        # base_url debe ser una URI (la ruta puede tener espacios). La hoja de estilos se pasa
        # explícita: si no existe falla, en vez de generar un PDF sin estilos.
        # El FontConfiguration es obligatorio para que se registren los @font-face (Public Sans) de
        # la hoja de estilos; sin él WeasyPrint usa en silencio la fuente del sistema.
        fuentes = FontConfiguration()
        return HTML(string=html, base_url=_DIRECTORIO.as_uri() + "/").write_pdf(
            stylesheets=[
                CSS(filename=str(_DIRECTORIO / nombre), font_config=fuentes)
                for nombre in ("informe.css", "portada.css")
            ],
            font_config=fuentes,
        )
