"""Fiestas patronales por distrito (262) — consolidación de cuatro fuentes secundarias con votación.

Fuentes (copias crudas en data/fuente_original/fuentes_externas/fiestas/):
  W   Wikipedia en español (API MediaWiki): artículo de cada distrito; se guarda la oración citada
  MDE municipiosdeelsalvador.com: 14 páginas departamentales + 262 fichas de distrito
  LN  lista nacional: elsalvadormipais.com (replicada en elsalvadorct.blogspot.com)
  CH  chalatenango.sv: lista departamental de Chalatenango
Reglas: un rango candidato por fuente; dos rangos coinciden si se solapan o distan <= 3 días; se elige el
rango con más fuentes coincidentes (desempate CH > MDE > W > LN). Confianza «alta» con >= 2 fuentes, «media» con 1.
Las decisiones manuales (alias de nombres, títulos de Wikipedia y cuatro correcciones verificadas) están
declaradas en las constantes de este módulo.

Salida: cfg.FIESTAS_FILE — una fila por distrito y día de fiesta, 2022-2026 (mismo formato que FERIADOS_FILE).
El detalle de la compilación (fuente de cada fecha) queda en cfg.EXTRACCION_DIR/fiestas/, no en el archivo de datos.
Lista de distritos y nombres: cfg.load_censo().
Licencia: el contenido derivado de Wikipedia es CC BY-SA 4.0.

Uso: python -m src.extraccion.fiestas [--refrescar]
"""
from __future__ import annotations
import re, sys, json, html, time, datetime as dt, unicodedata
import pandas as pd
from src import config as cfg
from ._web import get, get_json, norm

MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
MS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
mnum = lambda s: (8 if norm(s) == "setiembre" else MS.index(norm(s))) + 1
SLUG_DEP = {"Ahuachapán": "ahuachapan", "Santa Ana": "santa-ana", "Sonsonate": "sonsonate", "Chalatenango": "chalatenango",
            "La Libertad": "la-libertad", "San Salvador": "san-salvador", "Cuscatlán": "cuscatlan", "La Paz": "la-paz",
            "Cabañas": "cabanas", "San Vicente": "san-vicente", "Usulután": "usulutan", "San Miguel": "san-miguel",
            "Morazán": "morazan", "La Unión": "la-union"}
DEP_DE_SLUG = {v: k for k, v in SLUG_DEP.items()}
WIKI_API = "https://es.wikipedia.org/w/api.php"
# Títulos fijados a mano: la búsqueda devolvía otra página (departamento, aeropuerto, club, homónimo)
WIKI_TITULOS = {("El Congo", "Santa Ana"): "El Congo", ("Mejicanos", "San Salvador"): "Mejicanos",
                ("Villa Dolores", "Cabañas"): "Dolores (El Salvador)", ("San Buena Ventura", "Usulután"): "San Buenaventura (El Salvador)",
                ("Ciudad Barrios", "San Miguel"): "Ciudad Barrios", ("San Antonio", "San Miguel"): "San Antonio (El Salvador)",
                ("San Salvador", "San Salvador"): "San Salvador", ("Soyapango", "San Salvador"): "Soyapango",
                ("Ilopango", "San Salvador"): "Ilopango", ("Metapán", "Santa Ana"): "Metapán"}
WIKI_MALAS = re.compile(r"Departamento|Aeropuerto|Club|Asociaci|Sitio de|Batalla|Universidad|Estadio|Diócesis|Volcán|Lago|Río", re.I)
MDE_SLUG_EXTRA = {"Delgado": ["ciudad-delgado"], "Ciudad Delgado": ["ciudad-delgado"], "Villa Dolores": ["dolores"], "San Buena Ventura": ["san-buenaventura"],
                  "San Antonio": ["san-antonio-del-mosco"], "Cancasque": ["san-jose-cancasque"], "Las Flores": ["san-jose-las-flores"],
                  "San José": ["san-jose-la-fuente"], "Victoria": ["victoria"]}
ALIAS = {"nueva san salvador": "santa tecla", "ciudad delgado": "delgado", "mercedes humana": "mercedes umana",
         "san buenaventura": "san buena ventura", "comapala": "comalapa", "dolores": "villa dolores",
         "san antonio del mosco": "san antonio", "san jose cancasque": "cancasque", "san jose las flores": "las flores",
         "san jose la fuente": "san jose", "sta rosa guachipilin": "santa rosa guachipilin", "ocoaitique": "jocoaitique",
         "san luis la herradura la paz": "san luis la herradura", "villa victoria": "victoria", "san fco gotera": "san francisco gotera",
         "fiestas patronales guacotecti": "guacotecti", "monte de san juan": "monte san juan", "san pedro tacachico": "san pablo tacachico",
         "mercedes de la ceiba": "mercedes la ceiba", "rosario de la paz": "el rosario", "fiestas patronales san salvador": "san salvador",
         "salcoatitlan": "salcoatitan", "fiestas patronales alegria": "alegria", "sto domingo de guzman": "santo domingo de guzman"}
# Correcciones verificadas a mano contra la cita (ver docs/fuentes_externas_metodologia.md)
WIKI_FECHA_MANUAL = {("Metapán", "Santa Ana"): (6, 24, 6, 28), ("Soyapango", "San Salvador"): (10, 1, 10, 12)}
NOMBRE = {"W": "Wikipedia (es)", "MDE": "municipiosdeelsalvador.com", "LN": "Lista nacional (elsalvadormipais.com)", "CH": "chalatenango.sv"}
PRIORIDAD = ["CH", "MDE", "W", "LN"]


def _texto(h: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "\n", h)).replace("\xa0", " ")


def _rango(s: str):
    r = re.search(rf"(\d{{1,2}})°?(?:\s+de\s+({MESES}))?\s*(?:al|y el|y|–|-|a|hasta el)\s*(\d{{1,2}})\s+de\s+({MESES})", s, re.I)
    if r:
        mf = mnum(r[4]); mi = mnum(r[2]) if r[2] else mf
        if mi == mf and int(r[3]) < int(r[1]):
            return None      # rango invertido (error tipográfico de la fuente): se descarta
        return (mi, int(r[1]), mf, int(r[3]))
    r = re.search(rf"(\d{{1,2}})\s+de\s+({MESES})", s, re.I)
    return (mnum(r[2]), int(r[1]), mnum(r[2]), int(r[1])) if r else None


def _patrono(s: str) -> str:
    m = re.search(r"en honor (?:de |a |al |del )?(?:la |el |los |las |su |sus )?(?:(?:santo |santa )?patron[oa]s?,? )?([A-ZÁÉÍÓÚÑ][^,\.;\(\n|]{2,50})", s)
    if not m:
        return ""
    p = re.split(r"\s+(?:que|y el|y la fiesta|cuyas|celebrad|se celebra|entre)\b|\d|=", m[1])[0]
    return p.strip(' ,;y"')


# ----------------------------------------------------------------------------------------------- Wikipedia
def _wikipedia(pob, cache, refrescar):
    arch = cache / "wikipedia_extractos.json"
    datos = {} if refrescar or not arch.exists() else json.loads(arch.read_text(encoding="utf-8"))

    def api(params):
        for i in range(8):
            try:
                j = get_json(WIKI_API, params=params)
                if j and "query" in j:
                    return j
            except Exception:
                pass
            time.sleep(5 + 5 * i)      # límite de tasa de la API: espera creciente
        raise RuntimeError("Wikipedia no responde")

    def extracto(titulo):
        pg = next(iter(api({"action": "query", "prop": "extracts", "explaintext": 1, "titles": titulo, "format": "json", "redirects": 1})["query"]["pages"].values()))
        return pg.get("title", titulo), pg.get("extract", "") or ""

    for _, r in pob.iterrows():
        if r.clave in datos:
            continue
        clave = (r.distrito, r.departamento)
        if clave in WIKI_TITULOS:
            titulo, txt = extracto(WIKI_TITULOS[clave])
        else:
            hits = [h["title"] for h in api({"action": "query", "list": "search", "srsearch": f"{r.distrito} {r.departamento} El Salvador",
                                             "srlimit": 3, "format": "json"})["query"]["search"]]
            titulo = next((h for h in hits if norm(r.distrito) in norm(h)), hits[0] if hits else None)
            titulo, txt = extracto(titulo) if titulo else (None, "")
            if titulo and (WIKI_MALAS.search(titulo) or "El Salvador" not in txt[:600]):
                for cand in [f"{r.distrito} (El Salvador)", f"{r.distrito} ({r.departamento})", r.distrito]:
                    t2, x2 = extracto(cand)
                    if x2 and "El Salvador" in x2[:600] and not WIKI_MALAS.search(t2):
                        titulo, txt = t2, x2; break
        datos[r.clave] = {"distrito": r.distrito, "departamento": r.departamento, "titulo": titulo, "texto": txt}
        arch.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8"); time.sleep(0.6)

    FP = re.compile(r"fiestas?\s+(titulares?\s+y\s+)?patronal|fiestas?\s+titular|fiestas?\s+(en\s+honor|julias|agostinas)|romer[ií]a|fiestas? (de|en honor)", re.I)
    PAT = re.compile(r"patron[oa]|en honor (a|al|de)", re.I)
    BAD = re.compile(r"decreto|fundad|erigi|título de|se le dio|nombre en honor", re.I)
    filas = []
    for _, r in pob.iterrows():
        o = datos[r.clave]; cand = []
        for s in [x.strip() for x in re.split(r"(?<=[\.\n])\s+", o["texto"]) if x.strip()]:
            if BAD.search(s) and not FP.search(s):
                continue
            fp, pt, f = bool(FP.search(s)), bool(PAT.search(s)), _rango(s)
            if fp and f: c = 4
            elif (fp and re.search(MESES, s, re.I)) or (pt and f): c = 3
            elif fp or pt: c = 1
            else: continue
            cand.append((c, s, f))
        cand.sort(key=lambda x: -x[0])
        c, s, f = cand[0] if cand else (0, "", None)
        conf = {4: "alta", 3: "media", 1: "baja", 0: "sin_dato"}[c]
        if conf == "baja" or (f and conf != "alta" and not FP.search(s) and not re.search(r"celebra|festej|fiesta|feria", s, re.I)):
            f = None; conf = "baja" if conf != "sin_dato" else conf
        if f and conf == "alta" and re.search(r"\b1[5-9]\d\d\b|\b20[0-2]\d\b", s):
            conf = "anio_en_cita"      # posible fecha histórica: no entra como candidata
        mes_solo = mnum(re.search(MESES, s, re.I)[0]) if conf == "media" and not f and re.search(MESES, s, re.I) else None
        filas.append({"clave": r.clave, "conf_w": conf, "fecha_w": f, "mes_w": mes_solo,
                      "patrono_w": _patrono(s), "cita_wikipedia": re.sub(r"\s+", " ", s)[:350],
                      "url_w": "https://es.wikipedia.org/wiki/" + (o["titulo"] or "").replace(" ", "_")})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------- listas externas
def _mde_departamentos(cache, refrescar):
    filas = []
    for dep, slug in SLUG_DEP.items():
        t = _texto(get(f"https://www.municipiosdeelsalvador.com/{slug}", cache / "mde" / f"{slug}.html", refrescar=refrescar))
        hs = [m.start() for m in re.finditer(r"\n\s*Fiestas\s+(?:patronales|en los municipios)[^\n]{0,60}\n", t)]
        hs = [h for h in hs if re.search(r"\d{1,2}\s*(?:al|y|\u2013|-)\s*\d", t[h:h + 600])]
        if not hs:
            continue      # Chalatenango: su página no tiene la sección; se usa chalatenango.sv
        sec = t[hs[0] + 1:hs[0] + 9000]
        fin = re.search(r"\n\s*(Comentarios|Deja un|Municipios de|Artículos relacionados|Ver también|Leer más)", sec[40:])
        sec = sec[:fin.start() + 40] if fin else sec
        ent, buf = [], ""
        for l in [re.sub(r"\s+", " ", l).strip() for l in sec.split("\n")][1:]:
            if not l: continue
            if re.match(r"^[,:–-]", l) or (buf and not re.search(r"\d|marzo o abril|domingo", buf)): buf = (buf + " " + l).strip()
            else:
                if buf: ent.append(buf)
                buf = l
        if buf: ent.append(buf)
        for e in ent:
            m = re.match(r"^\s*(?:\d+\.\s*)?([A-ZÁÉÍÓÚÑ][^,:\d\(]{2,45}?)\s*(?:\(cabecera[^)]*\))?\s*(?:[,:]|(?=del?\s+\d))\s*(.*)$", e)
            if not m or re.match(r"^(Dr\.|Luego)", m[1]): continue
            f = _rango(m[2])
            if f: filas.append({"departamento": dep, "nombre": m[1].strip(), "fecha": f, "patrono": _patrono(m[2]),
                                "url": f"https://www.municipiosdeelsalvador.com/{slug}"})
    return pd.DataFrame(filas).drop_duplicates(["departamento", "nombre"])


def _mde_distritos(pob, cache, refrescar):
    arch = cache / "mde_fichas_distrito.json"
    datos = {} if refrescar or not arch.exists() else json.loads(arch.read_text(encoding="utf-8"))
    for _, r in pob.iterrows():
        if r.clave in datos: continue
        slug = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", r.distrito).encode("ascii", "ignore").decode().lower()).strip("-")
        got = {"url": "", "texto": ""}
        for s in MDE_SLUG_EXTRA.get(r.distrito, []) + [slug]:
            url = f"https://www.municipiosdeelsalvador.com/{SLUG_DEP[r.departamento]}/{s}"
            h = get(url)
            if h and "Cultura" in h:
                lin = [re.sub(r"\s+", " ", l).strip() for l in _texto(h).split("\n") if re.search(r"(?i)fiestas? (patronal|titular)", l)]
                got = {"url": url, "texto": lin[0] if lin else ""}; break
            time.sleep(0.4)
        datos[r.clave] = got; arch.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8"); time.sleep(0.4)
    d = pd.DataFrame([{"clave": k, **v} for k, v in datos.items()])
    d = d[d.texto != ""]
    d = d[~d.texto.duplicated(keep=False)]      # el sitio sirve la misma ficha a distritos homónimos: se descartan
    d["fecha"] = d.texto.map(_rango); d["patrono"] = d.texto.map(_patrono)
    return d[d.fecha.notna()]


def _lista_nacional(cache, refrescar):
    AB = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}
    MM = [m.upper() for m in MS]; url = "https://www.elsalvadormipais.com/fiestas-patronales-de-el-salvador"
    L = [l.strip() for l in _texto(get(url, cache / "elsalvadormipais.html", refrescar=refrescar)).split("\n") if l.strip()]
    k, mes, filas = L.index("ENERO"), None, []
    while k < len(L):
        l = L[k]
        if l in MM: mes = MM.index(l) + 1; k += 1; continue
        m = re.match(r"^(\d{1,2})(?:\s*([A-Za-z]{3})\.?)?\s*[-–]\s*(\d{1,2})(?:\s*([A-Za-z]{3})\.?)?$", l)
        if m and mes and k + 2 < len(L):
            lugar, santo = L[k + 1], L[k + 2]
            mi = AB[m[2].lower()] if m[2] else mes; mf = AB[m[4].lower()] if m[4] else mes
            dep = next((d for d in SLUG_DEP if lugar.rstrip().endswith(d)), "")
            nombre = lugar[:len(lugar) - len(dep)].strip(" ,") if dep else lugar
            if nombre == "Quezaltepeque" and dep == "Chalatenango": nombre = "Concepción Quezaltepeque"
            filas.append({"departamento": dep, "nombre": nombre, "fecha": (mi, int(m[1]), mf, int(m[3])), "patrono": santo, "url": url})
            k += 3; continue
        if l.startswith(("Comentarios", "9 comentarios", "Deja una")): break
        k += 1
    return pd.DataFrame(filas).drop_duplicates(["departamento", "nombre"])


def _chalatenango(cache, refrescar):
    url = "https://chalatenango.sv/fiestas-en-los-municipios-de-chalatenango"
    t = _texto(get(url, cache / "chalatenango_sv.html", refrescar=refrescar))
    sec = re.sub(r":\s*\n\s*", ": ", t[t.find("ordenadas por municipio"):t.find("ordenadas por mes", t.find("ordenadas por municipio"))])
    filas = []
    for l in sec.split("\n"):
        m = re.match(r"^\s*(?:\d+\.\s*)?([A-ZÁÉÍÓÚÑ][^:]{2,40}):\s*(.+)$", l.strip())
        if m and _rango(m[2]):
            filas.append({"departamento": "Chalatenango", "nombre": m[1].strip().replace("Comapala", "Comalapa"),
                          "fecha": _rango(m[2]), "patrono": _patrono(m[2]), "url": url})
    return pd.DataFrame(filas).drop_duplicates("nombre")


# ------------------------------------------------------------------------------------------ consolidación
def _clave(nombre, dep):
    n = norm(nombre)
    if n == "san rafael" and norm(dep) == "san miguel": n = "san rafael oriente"
    return ALIAS.get(n, n) + "|" + norm(dep)


def _intervalo(f):
    mi, di, mf, df = [int(x) for x in f]
    return dt.date(2025, mi, min(di, 28)), dt.date(2025 if mf >= mi else 2026, mf, min(df, 28))


def _coinciden(a, b):
    (s1, e1), (s2, e2) = _intervalo(a), _intervalo(b)
    return (min(e1, e2) - max(s1, s2)).days >= -3


def main(refrescar: bool = False) -> pd.DataFrame:
    cache = cfg.EXTRACCION_DIR / "fiestas"; (cache / "mde").mkdir(parents=True, exist_ok=True)
    pob = cfg.load_censo()[["departamento", "municipio", "distrito"]].copy()
    pob["clave"] = pob.departamento + "|" + pob.distrito
    W = _wikipedia(pob, cache, refrescar)
    idx = {}
    for s, d in [("MDE", _mde_departamentos(cache, refrescar)), ("LN", _lista_nacional(cache, refrescar)), ("CH", _chalatenango(cache, refrescar))]:
        for _, r in d.iterrows():
            idx.setdefault(_clave(r.nombre, r.departamento), {})[s] = (r.fecha, r.url, r.patrono)
    mdd = _mde_distritos(pob, cache, refrescar).set_index("clave")
    filas = []
    for _, r in pob.merge(W, on="clave").iterrows():
        cands = dict(idx.get(_clave(r.distrito, r.departamento), {}))
        if "MDE" not in cands and r.clave in mdd.index:      # ficha de distrito como respaldo (mismo sitio)
            x = mdd.loc[r.clave]; cands["MDE"] = (x.fecha, x.url, x.patrono)
        manual = WIKI_FECHA_MANUAL.get((r.distrito, r.departamento))
        if manual: cands["W"] = (manual, r.url_w, r.patrono_w)
        elif r.conf_w == "alta": cands["W"] = (r.fecha_w, r.url_w, r.patrono_w)
        if cands:
            score = {s: sum(_coinciden(v[0], w[0]) for t, w in cands.items() if t != s) for s, v in cands.items()}
            best = max(cands, key=lambda s: (score[s], -PRIORIDAD.index(s)))
            f = cands[best][0]
            sup = [s for s in cands if s == best or _coinciden(cands[s][0], f)]; dis = [s for s in cands if s not in sup]
            conf = "alta" if len(sup) >= 2 else "media"
            nota = "Coinciden: " + ", ".join(NOMBRE[s] for s in sup)
            if dis: nota += "; discrepa: " + ", ".join(f"{NOMBRE[s]} {c[1]}/{c[0]}–{c[3]}/{c[2]}" for s in dis for c in [cands[s][0]])
            patrono = r.patrono_w or next((cands[s][2] for s in sup if cands[s][2]), "")
            fila = dict(f=f, conf=conf, n=len(sup), fuentes="; ".join(NOMBRE[s] for s in sup), urls=" ; ".join(str(cands[s][1]) for s in sup), nota=nota, patrono=patrono)
        elif r.mes_w:
            fila = dict(f=(r.mes_w, None, r.mes_w, None), conf="baja", n=1, fuentes=NOMBRE["W"], urls=r.url_w, nota="Solo mes, sin días", patrono=r.patrono_w)
        else:
            fila = dict(f=(None,) * 4, conf="sin_dato", n=0, fuentes="", urls="", nota="", patrono=r.patrono_w)
        filas.append({"departamento": r.departamento, "municipio": r.municipio, "distrito": r.distrito,
                      "patrono": fila["patrono"], "mes_inicio": fila["f"][0], "dia_inicio": fila["f"][1], "mes_fin": fila["f"][2],
                      "dia_fin": fila["f"][3], "regla_movil": "", "confianza": fila["conf"], "n_fuentes_coinciden": fila["n"],
                      "fuentes": fila["fuentes"], "fuente_url": fila["urls"], "cita_wikipedia": r.cita_wikipedia, "nota": fila["nota"]})
    o = pd.DataFrame(filas)

    def fijar(dist, dep, valores, conf, fuentes, urls, nota, regla=""):
        i = (o.distrito == dist) & (o.departamento == dep); assert i.sum() == 1, dist
        for k, v in valores.items(): o.loc[i, k] = v
        o.loc[i, ["confianza", "fuentes", "fuente_url", "nota", "regla_movil", "n_fuentes_coinciden"]] = [conf, fuentes, urls, nota, regla, len(fuentes.split(";"))]
    # Búsqueda dirigida (sin ficha válida en las fuentes sistemáticas)
    fijar("El Rosario", "Cuscatlán", {"mes_inicio": 9, "dia_inicio": 29, "mes_fin": 10, "dia_fin": 8, "patrono": "Virgen del Rosario"}, "media",
          "municipiosdeelsalvador.com; EcuRed", "https://www.municipiosdeelsalvador.com/cuscatlan/el-rosario ; https://www.ecured.cu/El_Rosario_(El_Salvador)",
          "Rango 29/9–8/10; EcuRed indica el primer domingo de octubre (contenido en el rango)")
    fijar("El Rosario", "Morazán", {"mes_inicio": 10, "dia_inicio": None, "mes_fin": 10, "dia_fin": None, "patrono": "Virgen del Rosario"}, "media",
          "Wikipedia (es); hpturismo.net; esa123.com", "https://es.wikipedia.org/wiki/El_Rosario_(Morazán) ; https://hpturismo.net/el_salvador_morazan/ ; https://esa123.com/tema-de-el-salvador.php?id_tema=8",
          "Fecha móvil: primer domingo de octubre", "primer domingo de octubre")
    for c in ["mes_inicio", "dia_inicio", "mes_fin", "dia_fin"]:
        o[c] = pd.to_numeric(o[c]).astype("Int64")
    o.to_csv(cache / "compilacion_fiestas_detalle.csv", index=False, encoding="utf-8-sig")   # trazabilidad interna
    cal = _calendario(o)
    cal.to_excel(cfg.FIESTAS_FILE, index=False)
    print(f"  fiestas: {o.confianza.value_counts().to_dict()} | {len(cal)} filas distrito-día en el calendario")
    return cal


def _calendario(o: pd.DataFrame) -> pd.DataFrame:
    """Un registro por distrito y día de fiesta, 2022-2026 (mismo formato que feriados_elsalvador_2022_2026.xlsx)."""
    import calendar
    dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    f = lambda y, m, d: dt.date(y, m, min(d, calendar.monthrange(y, m)[1]))
    filas = []
    for _, r in o.iterrows():
        for y in range(2021, 2027):
            if r.regla_movil == "primer domingo de octubre":
                ini = dt.date(y, 10, 1); ini += dt.timedelta((6 - ini.weekday()) % 7); fin = ini
            else:
                ini = f(y, int(r.mes_inicio), int(r.dia_inicio))
                fin = f(y if int(r.mes_fin) >= int(r.mes_inicio) else y + 1, int(r.mes_fin), int(r.dia_fin))
            d = ini
            while d <= fin:
                if dt.date(2022, 1, 1) <= d <= dt.date(2026, 12, 31):
                    filas.append({"anio": d.year, "fecha": pd.Timestamp(d), "mes": d.month, "dia": d.day, "dia_semana": dias[d.weekday()],
                                  "departamento": r.departamento, "municipio": r.municipio, "distrito": r.distrito,
                                  "patrono": r.patrono if isinstance(r.patrono, str) and r.patrono else None})
                d += dt.timedelta(1)
    cal = pd.DataFrame(filas).drop_duplicates(["fecha", "departamento", "distrito"])
    cal = cal.sort_values(["fecha", "departamento", "distrito"]).reset_index(drop=True)
    assert cal.groupby(["departamento", "distrito"]).ngroups == 262, "distritos sin fechas en el calendario"
    return cal


if __name__ == "__main__":
    main(refrescar="--refrescar" in sys.argv)
