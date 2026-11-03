"""Mapa de distritos para la portada del informe por periodo (SVG generado en Python, sin dependencias nuevas).

Uso desde el generador del PDF:
    svg = mapa_distritos_svg(geojson, valores, etiquetas=4)
donde `geojson` es el GeoJSON de distritos del censo 2024 (propiedades Dep_min/Dis_min) y `valores` es {(departamento, distrito): siniestros_esperados}
para los distritos del informe. Los distritos sin valor se dibujan como fuera del alcance.
"""
from __future__ import annotations
import math, re, unicodedata
from xml.sax.saxutils import escape

ESCALA = ["#DCE6F0", "#A9C1D9", "#6F95BC", "#3A6A9B", "#12355B"]   # de menos a más siniestros esperados
FUERA = {"fill": "#F3F4F6", "stroke": "#CDD2D9"}
ALIAS = {"delgado": "ciudad delgado"}   # la capa oficial nombra "Delgado" a Ciudad Delgado
LAT0 = 13.7


def _norm(x: str) -> str:
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()
    return ALIAS.get(s, s)


def _proj(lon, lat):
    return (lon + 90.2) * math.cos(math.radians(LAT0)), -(lat - 14.5)


def cortes_cuantiles(valores, k=5):
    """Cortes por cuantiles, sin repetir: con pocos valores distintos hay menos clases (p. ej. un
    solo distrito = una clase), en vez de cinco clases idénticas."""
    v = sorted(valores)
    cortes = [v[min(len(v) - 1, int(round(len(v) * i / k)))] for i in range(1, k)]
    unicos = []
    for c in cortes:
        if c < v[-1] and (not unicos or c > unicos[-1]):
            unicos.append(c)
    return unicos


def _clase(x, cortes):
    return sum(x > c for c in cortes)


def _color_clase(indice, n_clases):
    """Color de la clase `indice` de `n_clases`: se reparten sobre la escala de claro a oscuro
    (la única clase usa el azul más oscuro)."""
    if n_clases <= 1:
        return ESCALA[-1]
    return ESCALA[round(indice * (len(ESCALA) - 1) / (n_clases - 1))]


TOLERANCIA = 0.45  # unidades del viewBox (de 1000 de ancho): ~0.08 mm en la portada impresa


def _aligerar(puntos, tolerancia=TOLERANCIA):
    """Descarta los vértices a menos de `tolerancia` del último conservado. La capa oficial trae
    ~735 mil vértices (9 MB de SVG y ~7 s de WeasyPrint); a esta escala la diferencia no se ve y
    el PDF sigue siendo vectorial. Conserva siempre el primero y el último vértice."""
    if len(puntos) <= 4:
        return puntos
    t2 = tolerancia * tolerancia
    kept = [puntos[0]]
    for x, y in puntos[1:-1]:
        kx, ky = kept[-1]
        if (x - kx) ** 2 + (y - ky) ** 2 >= t2:
            kept.append((x, y))
    kept.append(puntos[-1])
    return kept


def posiciones_rotulos(centros_x, anchos, ancho_total, separacion=14.0):
    """Posición x de cada rótulo (en el orden dado, que debe ser el de `centros_x` ascendente):
    cada uno lo más cerca posible de su distrito, sin superponerse con el anterior y sin salirse
    de [0, ancho_total]."""
    xs, previo = [], None
    for cx, w in zip(centros_x, anchos):
        x = cx if previo is None else max(cx, previo[0] + previo[1] + separacion)
        xs.append(x)
        previo = (x, w)
    if xs and xs[-1] + anchos[-1] > ancho_total:  # se sale por la derecha: se corre hacia la izquierda
        xs[-1] = ancho_total - anchos[-1]
        for i in range(len(xs) - 2, -1, -1):
            xs[i] = min(xs[i], xs[i + 1] - anchos[i] - separacion)
    return xs


def ancho_rotulo(nombre, valor_texto):
    """Ancho aproximado (unidades del viewBox) del rótulo: el mayor de sus dos líneas."""
    return max(len(nombre) * 17 * 0.6, len(valor_texto) * 15 * 0.56)


def _redondo(x):
    return f"{x:,.0f}" if x >= 10 else f"{x:,.1f}"


def mapa_distritos_svg(geojson, valores: dict, etiquetas: int = 4, ancho: float = 1000.0) -> tuple[str, list]:
    val = {(_norm(d), _norm(n)): float(v) for (d, n), v in valores.items()}
    cortes = cortes_cuantiles(list(val.values()))
    feats = []
    for f in geojson["features"]:
        p = f["properties"]; g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        clave = (_norm(p["Dep_min"]), _norm(p["Dis_min"]))
        anillos = [[_proj(x, y) for x, y in poly[0]] for poly in polys]
        feats.append((clave, p["Dis_min"], anillos))
    xs = [x for _, _, a in feats for r in a for x, _ in r]; ys = [y for _, _, a in feats for r in a for _, y in r]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    s = ancho / (x1 - x0); alto = (y1 - y0) * s
    T = lambda pt: ((pt[0] - x0) * s, (pt[1] - y0) * s)

    def path(anillos):
        out = []
        for r in anillos:
            pts = _aligerar([T(p) for p in r])
            out.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
        return "".join(out)

    partes_fuera, partes_dentro, centros = [], [], {}
    for clave, nombre, anillos in feats:
        d = path(anillos)
        if clave in val:
            c = _color_clase(_clase(val[clave], cortes), len(cortes) + 1)
            partes_dentro.append(f'<path d="{d}" fill="{c}" stroke="#FFFFFF" stroke-width="0.6"/>')
            # centro aproximado: promedio de vertices del anillo mas grande
            r = max(anillos, key=len); pts = [T(p) for p in r]
            centros[clave] = (nombre, sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts), val[clave])
        else:
            partes_fuera.append(f'<path d="{d}" fill="{FUERA["fill"]}" stroke="{FUERA["stroke"]}" stroke-width="0.5"/>')

    # etiquetas en una fila sobre el mapa, separadas para no superponerse, con linea guia en codo
    margen = 92
    top = sorted(sorted(centros.values(), key=lambda t: -t[3])[:etiquetas], key=lambda t: t[1])
    rotulos = []
    ty = -margen + 26
    codo = -14
    textos_valor = [f"{_redondo(v)} esperados" for *_, v in top]
    anchos = [ancho_rotulo(n, t) for (n, *_), t in zip(top, textos_valor)]
    lxs = posiciones_rotulos([cx for _, cx, _, _ in top], anchos, ancho)
    for (nombre, cx, cy, v), lx, texto_valor in zip(top, lxs, textos_valor):
        rotulos.append(
            f'<polyline points="{cx:.1f},{cy:.1f} {cx:.1f},{codo:.1f} {lx:.1f},{codo - 12:.1f} {lx:.1f},{ty + 22:.1f}" '
            f'fill="none" stroke="#14202E" stroke-width="0.9"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.8" fill="#14202E"/>'
            f'<text x="{lx:.1f}" y="{ty - 2:.1f}" class="n">{escape(nombre)}</text>'
            f'<text x="{lx:.1f}" y="{ty + 16:.1f}" class="v">{texto_valor}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-margen} {ancho:.0f} {alto + margen:.0f}" '
           f'role="img" aria-label="Mapa de siniestros esperados por distrito">'
           f'<style>.n{{font-family:"Public Sans";font-weight:600;font-size:17px;fill:#14202E}}'
           f'.v{{font-family:"Public Sans";font-size:15px;fill:#4A5565}}</style>'
           f'{"".join(partes_fuera)}{"".join(partes_dentro)}{"".join(rotulos)}</svg>')
    leyenda = []
    lim = [min(val.values()), *cortes, max(val.values())]
    n = len(cortes) + 1
    for i in range(n):
        desde, hasta = _redondo(lim[i]), _redondo(lim[i + 1])
        leyenda.append({"color": _color_clase(i, n), "desde": desde, "hasta": hasta,
                        "etiqueta": desde if desde == hasta else f"{desde} a {hasta}"})
    return svg, leyenda
