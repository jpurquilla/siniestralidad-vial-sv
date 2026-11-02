"""Utilidades de descarga para los scripts de extracción (no las usan los notebooks).

Las rutas salen de src/config.py. Cada descarga se guarda en cfg.EXTRACCION_DIR: si la copia
existe se reutiliza, de modo que los scripts dan el mismo resultado aunque la página cambie.
Para volver a descargar, borrar la copia o usar --refrescar.
"""
from __future__ import annotations
import re, time, json, unicodedata
from pathlib import Path
import requests

SESION = requests.Session()
SESION.headers["User-Agent"] = "Mozilla/5.0 (siniestralidad-vial-sv, UES; uso academico)"


def norm(x) -> str:
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def get(url: str, copia: Path | None = None, binario: bool = False, refrescar: bool = False, **kw):
    if copia is not None and copia.exists() and not refrescar:
        return copia.read_bytes() if binario else copia.read_text(encoding="utf-8")
    error = None
    for i in range(5):
        try:
            r = SESION.get(url, timeout=120, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            dato = r.content if binario else r.text
            if copia is not None:
                copia.parent.mkdir(parents=True, exist_ok=True)
                copia.write_bytes(dato) if binario else copia.write_text(dato, encoding="utf-8")
            return dato
        except Exception as e:
            error = e; time.sleep(3 * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}: {error}")


def get_json(url: str, copia: Path | None = None, refrescar: bool = False, **kw):
    t = get(url, copia, refrescar=refrescar, **kw)
    return json.loads(t) if t is not None else None
