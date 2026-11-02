"""Extrae el parque vehicular del Observatorio Nacional de Seguridad Vial (ONASEVI/FONAT).

La fuente no ofrece archivo descargable: publica un tablero Power BI en
https://observatoriovial.fonat.gob.sv/parque-vehicular/
Este script consulta el endpoint público `querydata` de ese tablero (datos agregados, sin datos
personales) y guarda los valores sin transformarlos:

  cfg.PARQUE_FILE           fecha_ref, cod_mun, cod_dis, cod_clase, vehiculos
                            (las dos columnas territoriales tal como vienen; las unifica cfg.load_parque_vehicular)
  cfg.PARQUE_CATALOGO_FILE  cod_distrito, distrito, departamento (nombres del registro, último corte)

Validación: motocicletas (clase 309) al 31 de diciembre de 2015-2025 contra la tabla oficial
«Parque vehicular por año» del Observatorio.
Respuestas crudas en cfg.EXTRACCION_DIR / "parque_vehicular".

Uso: python -m src.extraccion.parque_vehicular [--refrescar]
"""
import sys, json, time, hashlib, datetime as dt
import pandas as pd
from src import config as cfg
from ._web import SESION

KEY = "c6a56b9d-136b-4adb-991b-258767fee73a"; MODELO = 1911560
URL = "https://wabi-paas-1-scus-api.analysis.windows.net/public/reports/querydata?synchronous=true"
CABECERA = {"X-PowerBI-ResourceKey": KEY, "Content-Type": "application/json"}
ENTIDAD = "01 PARQUE VEHICULAR"
MOTOS_OFICIAL = {2015: 167254, 2016: 205426, 2017: 246179, 2018: 293531, 2019: 336629, 2020: 371970,
                 2021: 428299, 2022: 490155, 2023: 552517, 2024: 619186, 2025: 687736}


def _col(p):
    return {"Column": {"Expression": {"SourceRef": {"Source": "p"}}, "Property": p}, "Name": p}


def _decodificar(resp):
    """Formato DSR comprimido de Power BI: diccionarios de valores + máscaras R (repetir) y Ø (nulo)."""
    ds = resp["results"][0]["result"]["data"]["dsr"]["DS"][0]
    dic = ds.get("ValueDicts", {}); ph = ds["PH"][0]
    filas = ph[[k for k in ph if k.startswith("DM")][0]]
    dn = [s.get("DN") for s in filas[0]["S"]]; n = len(dn); out, prev = [], [None] * n
    for r in filas:
        it = iter(r.get("C", [])); R = r.get("R", 0); N = r.get("Ø", 0); vals = []
        for i in range(n):
            if R >> i & 1: v = prev[i]
            elif N >> i & 1: v = None
            else:
                v = next(it)
                if dn[i] and isinstance(v, int): v = dic[dn[i]][v]
            vals.append(v)
        prev = vals; out.append(vals)
    return out


def _consulta(grupos, carpeta, fecha=None, refrescar=False):
    sel = [_col(g) for g in grupos] + [{"Aggregation": {"Expression": {"Column": {"Expression": {"SourceRef": {"Source": "p"}},
                                        "Property": "T.VEHICULOS"}}, "Function": 0}, "Name": "suma"}]
    q = {"Version": 2, "From": [{"Name": "p", "Entity": ENTIDAD, "Type": 0}], "Select": sel}
    if fecha:
        q["Where"] = [{"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "p"}}, "Property": "FECHA_REF"}}],
                                            "Values": [[{"Literal": {"Value": f"datetime'{fecha}T00:00:00'"}}]]}}}]
    cuerpo = {"version": "1.0.0", "queries": [{"Query": {"Commands": [{"SemanticQueryDataShapeCommand": {"Query": q,
              "Binding": {"Primary": {"Groupings": [{"Projections": list(range(len(sel)))}]},
                          "DataReduction": {"DataVolume": 4, "Primary": {"Window": {"Count": 30000}}}, "Version": 1}}}]}}],
              "cancelQueries": [], "modelId": MODELO}
    copia = carpeta / f"{fecha or 'todas'}_{hashlib.md5(json.dumps(cuerpo, sort_keys=True).encode()).hexdigest()[:10]}.json"
    if copia.exists() and not refrescar:
        return _decodificar(json.loads(copia.read_text()))
    for i in range(5):
        try:
            r = SESION.post(URL, headers=CABECERA, data=json.dumps(cuerpo), timeout=120); r.raise_for_status()
            copia.write_text(r.text); time.sleep(0.8)
            return _decodificar(r.json())
        except Exception as e:
            error = e; time.sleep(3 * (i + 1))
    raise error


def main(refrescar=False):
    carpeta = cfg.EXTRACCION_DIR / "parque_vehicular"; carpeta.mkdir(parents=True, exist_ok=True)
    fechas = sorted(dt.datetime.fromtimestamp(f / 1000, dt.timezone.utc).date().isoformat()
                    for f, _ in _consulta(["FECHA_REF"], carpeta, refrescar=refrescar))
    filas = []
    for f in fechas:
        filas += [(f, m, d, c, v) for m, d, c, v in _consulta(["COD.MUN.", "C.DIS", "COD.CLASE"], carpeta, f, refrescar)]
    df = pd.DataFrame(filas, columns=["fecha_ref", "cod_mun", "cod_dis", "cod_clase", "vehiculos"])
    for c in ["cod_mun", "cod_dis", "cod_clase", "vehiculos"]:
        df[c] = pd.to_numeric(df[c]).astype("Int64")
    motos = df[(df.cod_clase == 309) & df.fecha_ref.str.endswith("12-31")].groupby("fecha_ref").vehiculos.sum()
    for f, v in motos.items():
        assert v == MOTOS_OFICIAL[int(f[:4])], f"Motocicletas {f}: {v} distinto de la cifra oficial"
    df.to_csv(cfg.PARQUE_FILE, index=False)

    cat = pd.DataFrame(_consulta(["C.DIS", "DISTRITO", "DEPARTAMENTO"], carpeta, fechas[-1], refrescar),
                       columns=["cod_distrito", "distrito", "departamento", "vehiculos"]).dropna(subset=["cod_distrito"])
    cat = cat.groupby("cod_distrito", as_index=False).agg(
        distrito=("distrito", lambda s: "; ".join(sorted({x for x in s if x}))), departamento=("departamento", "first"))
    cat["cod_distrito"] = cat.cod_distrito.astype(int)
    assert len(cat) == 262
    cat.to_csv(cfg.PARQUE_CATALOGO_FILE, index=False)
    print(f"OK: {len(fechas)} cortes, {len(df):,} filas; motocicletas 2015-2025 validadas; catálogo de {len(cat)} códigos")


if __name__ == "__main__":
    main(refrescar="--refrescar" in sys.argv)
