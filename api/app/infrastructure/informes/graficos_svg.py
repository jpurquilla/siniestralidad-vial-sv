"""Gráficos del informe como cadenas SVG puras (sin librerías de gráficos, sin PNG).

Cada función devuelve un `<svg>` autónomo con `viewBox` (escala al ancho del contenedor),
`role="img"`, `<title>` y `<desc>` (texto alternativo) y fuente Public Sans. Los colores de las
causas vienen siempre de `colores` (la fuente única `colores_causas` del informe).

Criterios de diseño: etiquetas directas al final de cada barra (sin leyendas separadas),
sin cuadrícula (solo el eje base), ejes con números redondos (pasos 1/2/2.5/5 × 10^k) y
texto con el contraste adecuado sobre cada color.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from xml.sax.saxutils import escape, quoteattr

from app.domain.informe import CausasDeFranja, FilaCausa, FilaDiaSemana, FilaDistritoFranja
from app.domain.models import Franja
from app.infrastructure.informes.formato import etiqueta_franja, numero

FUENTE = "'Public Sans', Arial, Helvetica, sans-serif"
AZUL = "#12355B"
TEXTO = "#14202E"
TEXTO_SUAVE = "#4A5565"
LINEA = "#D5DAE1"
ANCHO = 640  # unidades del viewBox; el SVG escala al ancho de la página

UMBRAL_ETIQUETA_SEGMENTO = 0.08  # el % solo se escribe dentro del segmento si mide más de 8%
_ANCHO_CARACTER = 0.57  # ancho medio de un carácter de Public Sans, en em (para medir etiquetas)


def _f(x: float) -> str:
    return f"{x:.1f}"


def _ancho_texto(texto: str, tam: float) -> float:
    return len(texto) * tam * _ANCHO_CARACTER


def _recortar(texto: str, ancho_px: float, tam: float) -> str:
    maximo = max(4, int(ancho_px / (tam * _ANCHO_CARACTER)))
    return texto if len(texto) <= maximo else texto[: maximo - 1] + "…"


def _texto(x: float, y: float, contenido: str, *, tam: float = 10, ancla: str = "start",
           color: str = TEXTO, peso: str = "normal") -> str:
    return (
        f'<text x="{_f(x)}" y="{_f(y)}" font-size="{tam}" text-anchor="{ancla}" fill="{color}" '
        f'font-weight="{peso}">{escape(contenido)}</text>'
    )


def _luminancia(hexcolor: str) -> float:
    def canal(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (canal(int(hexcolor[i : i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def color_texto_sobre(hexcolor: str) -> str:
    """Blanco u oscuro, el que dé mayor contraste (WCAG) sobre `hexcolor`."""
    lum = _luminancia(hexcolor)
    contraste_blanco = 1.05 / (lum + 0.05)
    contraste_oscuro = (lum + 0.05) / (_luminancia(TEXTO) + 0.05)
    return "#ffffff" if contraste_blanco >= contraste_oscuro else TEXTO


def _envolver(alto: float, titulo: str, descripcion: str, cuerpo: list[str], ancho: float = ANCHO) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(ancho)} {_f(alto)}" '
        f'role="img" aria-label={quoteattr(titulo)} font-family="{FUENTE}">'
        f"<title>{escape(titulo)}</title><desc>{escape(descripcion)}</desc>"
        + "".join(cuerpo)
        + "</svg>"
    )


def escala_agradable(maximo: float, marcas: int = 5) -> tuple[float, float]:
    """(tope, paso) con números redondos (pasos 1, 2, 2.5, 5 × 10^k): el tope es múltiplo del
    paso y >= maximo; da ~`marcas` marcas (0, 100, 200, 300… / 0, 50, 100…)."""
    if maximo <= 0:
        return 1.0, 0.25
    bruto = maximo / marcas
    magnitud = 10 ** math.floor(math.log10(bruto))
    paso = next(m * magnitud for m in (1, 2, 2.5, 5, 10) if m * magnitud >= bruto - 1e-12)
    return round(paso * math.ceil(maximo / paso - 1e-9), 10), paso


def _marcas(tope: float, paso: float) -> list[float]:
    return [round(i * paso, 10) for i in range(int(round(tope / paso)) + 1)]


def _etiqueta_marca(v: float, paso: float) -> str:
    return numero(v, 0 if float(paso).is_integer() else 1)


def _decimales(maximo: float) -> int:
    """Decimales de las etiquetas de valor: enteros si la escala es grande."""
    return 0 if maximo >= 20 else 1


def _eje_horizontal(x0: float, x1: float, y_base: float, tope: float, paso: float,
                    formato=None) -> list[str]:
    """Eje base con marcas pequeñas y números redondos (sin cuadrícula)."""
    formato = formato or (lambda v: _etiqueta_marca(v, paso))
    partes = [f'<line x1="{_f(x0)}" y1="{_f(y_base)}" x2="{_f(x1)}" y2="{_f(y_base)}" stroke="{LINEA}" stroke-width="1"/>']
    for v in _marcas(tope, paso):
        x = x0 + (x1 - x0) * v / tope
        partes.append(f'<line x1="{_f(x)}" y1="{_f(y_base)}" x2="{_f(x)}" y2="{_f(y_base + 3)}" stroke="{TEXTO_SUAVE}" stroke-width="0.8"/>')
        partes.append(_texto(x, y_base + 14, formato(v), tam=10, ancla="middle", color=TEXTO_SUAVE))
    return partes


# ----------------------------------------------------------------------------- Figura 1


def barras_causas_con_rango(filas: Sequence[FilaCausa], colores: dict[str, str]) -> str:
    """Barras horizontales del esperado por causa con su rango del 90% (línea con topes). El
    nombre va a la izquierda; el valor y el rango, directamente al final de cada barra."""
    tam_fila, x0, top = 28.0, 205.0, 6.0
    x1 = ANCHO - 120.0  # deja sitio a la etiqueta «1,234 (1,100–1,300)» tras la barra
    tope, paso = escala_agradable(max((f.ic90_sup for f in filas), default=0), 5)
    tope = max(tope, max((f.esperado for f in filas), default=0.0), 1e-9)
    alto_graf = top + len(filas) * tam_fila
    dec = _decimales(max((f.esperado for f in filas), default=0.0))  # 1 decimal si la escala es pequeña
    partes = [f'<line x1="{_f(x0)}" y1="{_f(top)}" x2="{_f(x0)}" y2="{_f(alto_graf)}" stroke="{LINEA}" stroke-width="1"/>']
    partes += _eje_horizontal(x0, x1, alto_graf, tope, paso)
    for i, f in enumerate(filas):
        y = top + i * tam_fila
        ym = y + tam_fila / 2
        ancho_barra = (x1 - x0) * f.esperado / tope
        partes.append(_texto(x0 - 8, ym + 4, f.causa, tam=11, ancla="end"))
        partes.append(
            f'<rect x="{_f(x0)}" y="{_f(y + 5)}" width="{_f(max(ancho_barra, 0.5))}" height="{_f(tam_fila - 10)}" '
            f'fill="{colores[f.causa]}"><title>{escape(f.causa)}: {numero(f.esperado, 0)} esperados '
            f"(rango 90%: {f.ic90_inf:,d}–{f.ic90_sup:,d})</title></rect>"
        )
        xi = x0 + (x1 - x0) * f.ic90_inf / tope
        xs = x0 + (x1 - x0) * f.ic90_sup / tope
        partes.append(f'<line x1="{_f(xi)}" y1="{_f(ym)}" x2="{_f(xs)}" y2="{_f(ym)}" stroke="{TEXTO}" stroke-width="1.2"/>')
        for xe in (xi, xs):
            partes.append(f'<line x1="{_f(xe)}" y1="{_f(ym - 4)}" x2="{_f(xe)}" y2="{_f(ym + 4)}" stroke="{TEXTO}" stroke-width="1.2"/>')
        partes.append(_texto(xs + 8, ym + 4, numero(f.esperado, dec), tam=10.5, peso="600"))
        partes.append(_texto(xs + 8 + _ancho_texto(numero(f.esperado, dec), 10.5) + 5, ym + 4,
                             f"({f.ic90_inf:,d}–{f.ic90_sup:,d})", tam=9.5, color=TEXTO_SUAVE))
    titulo = "Siniestros esperados por causa en el periodo"
    desc = "; ".join(f"{f.causa}: {numero(f.esperado, 0)} (rango 90% {f.ic90_inf:,d}–{f.ic90_sup:,d})" for f in filas)
    return _envolver(alto_graf + 22, titulo, desc, partes)


# ----------------------------------------------------------------------------- Figura 2


def _clave_compacta(causas: Sequence[str], colores: dict[str, str], y: float) -> tuple[list[str], float]:
    """Clave horizontal compacta (muestra de color + nombre) que fluye en filas, en el orden de
    los segmentos. Va pegada al gráfico y dentro del mismo SVG: es parte del gráfico."""
    partes: list[str] = []
    x, fila, tam, alto_fila = 10.0, 0, 10.0, 16.0
    for causa in causas:
        ancho = 14 + _ancho_texto(causa, tam) + 14
        if x + ancho > ANCHO - 10 and x > 10:
            x, fila = 10.0, fila + 1
        yy = y + fila * alto_fila
        partes.append(f'<rect x="{_f(x)}" y="{_f(yy - 8)}" width="9" height="9" rx="1.5" fill="{colores[causa]}"/>')
        partes.append(_texto(x + 13, yy + 0.5, causa, tam=tam))
        x += ancho
    return partes, (fila + 1) * alto_fila


def barras_composicion_por_franja(composicion: Sequence[CausasDeFranja], colores: dict[str, str]) -> str:
    """Barras apiladas al 100%, una por franja, con el % dentro del segmento solo si mide más
    de 8%. Las causas se identifican con una clave compacta pegada bajo las barras."""
    tam_fila, x0, top = 34.0, 125.0, 4.0
    x1 = ANCHO - 12.0
    alto_graf = top + len(composicion) * tam_fila
    partes = [f'<line x1="{_f(x0)}" y1="{_f(top)}" x2="{_f(x0)}" y2="{_f(alto_graf)}" stroke="{LINEA}" stroke-width="1"/>']
    partes += _eje_horizontal(x0, x1, alto_graf, 100.0, 25.0, formato=lambda v: f"{v:.0f}%")
    causas: list[str] = []
    for cf in composicion:
        for c in cf.causas:
            if c.causa not in causas:
                causas.append(c.causa)
    for i, cf in enumerate(composicion):
        y = top + i * tam_fila
        partes.append(_texto(x0 - 8, y + tam_fila / 2 + 4, etiqueta_franja(cf.franja), tam=11, ancla="end"))
        acumulado = 0.0
        for c in cf.causas:
            ancho = (x1 - x0) * c.proporcion
            xs = x0 + (x1 - x0) * acumulado
            acumulado += c.proporcion
            if ancho <= 0:
                continue
            partes.append(
                f'<rect x="{_f(xs)}" y="{_f(y + 4)}" width="{_f(ancho)}" height="{_f(tam_fila - 8)}" '
                f'fill="{colores[c.causa]}" stroke="#ffffff" stroke-width="0.8">'
                f"<title>{escape(etiqueta_franja(cf.franja))} · {escape(c.causa)}: {c.proporcion * 100:.1f}%</title></rect>"
            )
            if c.proporcion > UMBRAL_ETIQUETA_SEGMENTO:
                partes.append(_texto(xs + ancho / 2, y + tam_fila / 2 + 3.6, f"{c.proporcion * 100:.0f}%",
                                     tam=10, ancla="middle", color=color_texto_sobre(colores[c.causa])))
    clave, alto_clave = _clave_compacta(causas, colores, alto_graf + 34)
    partes.extend(clave)
    titulo = "Composición de causas por franja horaria"
    desc = " | ".join(
        etiqueta_franja(cf.franja) + ": " + ", ".join(f"{c.causa} {c.proporcion * 100:.1f}%" for c in cf.causas)
        for cf in composicion
    )
    return _envolver(alto_graf + 34 + alto_clave, titulo, desc, partes)


# ----------------------------------------------------------------------------- Figura 3


def barras_dia_semana(filas: Sequence[FilaDiaSemana]) -> str:
    """Barras verticales del esperado PROMEDIO por día, de lunes a domingo, con el valor sobre
    cada barra."""
    x0, x1, top, alto_barras = 44.0, ANCHO - 12.0, 24.0, 130.0
    maximo = max((f.esperado for f in filas), default=0.0)
    tope, paso = escala_agradable(maximo, 4)
    base = top + alto_barras
    partes = [f'<line x1="{_f(x0)}" y1="{_f(base)}" x2="{_f(x1)}" y2="{_f(base)}" stroke="{LINEA}" stroke-width="1"/>']
    for v in _marcas(tope, paso):
        yy = base - alto_barras * v / tope
        partes.append(f'<line x1="{_f(x0 - 3)}" y1="{_f(yy)}" x2="{_f(x0)}" y2="{_f(yy)}" stroke="{TEXTO_SUAVE}" stroke-width="0.8"/>')
        partes.append(_texto(x0 - 6, yy + 3.5, _etiqueta_marca(v, paso), tam=10, ancla="end", color=TEXTO_SUAVE))
    partes.append(f'<line x1="{_f(x0)}" y1="{_f(top)}" x2="{_f(x0)}" y2="{_f(base)}" stroke="{LINEA}" stroke-width="1"/>')
    decimales = _decimales(maximo)
    ancho_col = (x1 - x0) / max(len(filas), 1)
    for i, f in enumerate(filas):
        cx = x0 + ancho_col * (i + 0.5)
        alto = alto_barras * f.esperado / tope
        partes.append(
            f'<rect x="{_f(cx - ancho_col * 0.3)}" y="{_f(base - alto)}" width="{_f(ancho_col * 0.6)}" height="{_f(alto)}" '
            f'fill="{AZUL}"><title>{escape(f.dia)}: {numero(f.esperado, 1)} esperados por día '
            f"(promedio de {f.veces} {'día' if f.veces == 1 else 'días'})</title></rect>"
        )
        partes.append(_texto(cx, base - alto - 5, numero(f.esperado, decimales), tam=10.5, ancla="middle", peso="600"))
        partes.append(_texto(cx, base + 16, f.dia, tam=11, ancla="middle"))
    titulo = "Siniestros esperados por día de la semana (promedio por día)"
    desc = "; ".join(f"{f.dia}: {numero(f.esperado, 1)} por día ({f.veces} veces en el periodo)" for f in filas)
    return _envolver(base + 26, titulo, desc, partes)


# ----------------------------------------------------------------------------- Figura 4


def _color_calor(t: float) -> str:
    """Interpola de #DCE6F0 (0) al azul institucional #12355B (1): los extremos de la escala de la portada."""
    inicio, fin = (220, 230, 240), (18, 53, 91)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(inicio, fin, strict=True))


def mapa_calor_distrito_franja(filas: Sequence[FilaDistritoFranja]) -> str:
    """Mapa de calor distrito × franja (una fila por distrito, valor dentro de cada celda). La
    escala de color llega a un tope redondo y se rotula con números redondos."""
    x0, ancho_col, tam_fila, top = 170.0, 110.0, 18.0, 34.0
    franjas = list(Franja)
    maximo = max((v for f in filas for v in f.franjas.values()), default=0.0)
    tope, paso = escala_agradable(maximo, 5)
    partes = []
    for j, fr in enumerate(franjas):
        cx = x0 + ancho_col * (j + 0.5)
        base, _, rango = etiqueta_franja(fr).partition(" (")
        partes.append(_texto(cx, top - 17, base, tam=10.5, ancla="middle", peso="600", color=TEXTO))
        partes.append(_texto(cx, top - 6, rango.rstrip(")") + " h", tam=9.5, ancla="middle", color=TEXTO_SUAVE))
    for i, f in enumerate(filas):
        y = top + i * tam_fila
        partes.append(_texto(x0 - 8, y + tam_fila / 2 + 3.6, _recortar(f.distrito, x0 - 14, 10.5), tam=10.5, ancla="end"))
        for j, fr in enumerate(franjas):
            v = f.franjas[fr]
            color = _color_calor(min(v / tope, 1.0))
            xs = x0 + ancho_col * j
            partes.append(
                f'<rect x="{_f(xs)}" y="{_f(y)}" width="{_f(ancho_col)}" height="{_f(tam_fila)}" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1.5"><title>{escape(f.distrito)} · {escape(etiqueta_franja(fr))}: '
                f"{numero(v)} esperados</title></rect>"
            )
            partes.append(_texto(xs + ancho_col / 2, y + tam_fila / 2 + 3.6, numero(v), tam=10, ancla="middle",
                                 color=color_texto_sobre(color)))
    y_fin = top + len(filas) * tam_fila
    # Escala: 20 pasos discretos (un degradado SVG no se pinta bien en PDF) con marcas redondas.
    ancho_escala, pasos = 4 * ancho_col, 20
    for k in range(pasos):
        partes.append(
            f'<rect x="{_f(x0 + k * ancho_escala / pasos)}" y="{_f(y_fin + 12)}" width="{_f(ancho_escala / pasos + 0.3)}" '
            f'height="8" fill="{_color_calor((k + 0.5) / pasos)}"/>'
        )
    for v in _marcas(tope, paso):
        x = x0 + ancho_escala * v / tope
        partes.append(f'<line x1="{_f(x)}" y1="{_f(y_fin + 20)}" x2="{_f(x)}" y2="{_f(y_fin + 23)}" stroke="{TEXTO_SUAVE}" stroke-width="0.8"/>')
        partes.append(_texto(x, y_fin + 34, _etiqueta_marca(v, paso), tam=10, ancla="middle", color=TEXTO_SUAVE))
    titulo = "Mapa de calor de siniestros esperados por distrito y franja horaria"
    desc = "; ".join(f"{f.distrito}: " + ", ".join(f"{etiqueta_franja(fr)} {numero(f.franjas[fr])}" for fr in franjas) for f in filas)
    return _envolver(y_fin + 42, titulo, desc, partes)
