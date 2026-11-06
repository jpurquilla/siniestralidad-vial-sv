"""Redacción del resumen del informe por periodo (párrafo en lenguaje llano + tabla de 4 filas).

Python puro y determinista: todas las cifras salen de los datos del informe, nada se escribe a
mano. Es la fuente única del texto: la web y el PDF lo muestran tal cual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from app.domain.informe import FilaDiaSemana, FilaResumen, ResumenInforme
from app.domain.models import Franja

MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)
NUM_DISTRITOS_TEXTO = 4

# Cantidad de distritos del estudio: con todos ellos el resumen habla del país.
_TOTAL_DISTRITOS = 103

_ETIQUETA_FRANJA = {
    Franja.MADRUGADA: "Madrugada (00-05)",
    Franja.MANANA: "Mañana (06-11)",
    Franja.TARDE: "Tarde (12-17)",
    Franja.NOCHE: "Noche (18-23)",
}
_SUSTANTIVO_FRANJA = {
    Franja.MADRUGADA: "madrugada", Franja.MANANA: "mañana", Franja.TARDE: "tarde", Franja.NOCHE: "noche",
}


def conteo(valor: float) -> str:
    """Entero con coma de miles («1,982»), redondeo half-up."""
    return f"{int(valor + 0.5):,d}"


def porcentaje_entero(parte: float, total: float) -> int:
    return int(parte / total * 100 + 0.5) if total > 0 else 0


def fecha_larga(f: date) -> str:
    return f"{f.day} de {MESES[f.month - 1]} de {f.year}"


def periodo_portada(inicio: date, fin: date) -> str:
    """Periodo para la portada del PDF: «2 de noviembre al 2 de diciembre de 2026»; del mismo mes,
    «2 al 30 de noviembre de 2026»; de años distintos, con ambos años; un solo día, «2 de noviembre de 2026»."""
    if inicio == fin:
        return fecha_larga(inicio)
    if inicio.year == fin.year and inicio.month == fin.month:
        return f"{inicio.day} al {fin.day} de {MESES[fin.month - 1]} de {fin.year}"
    if inicio.year == fin.year:
        return f"{inicio.day} de {MESES[inicio.month - 1]} al {fecha_larga(fin)}"
    return f"{fecha_larga(inicio)} al {fecha_larga(fin)}"


def frase_distritos_principales(distritos: Sequence[tuple[str, float]], total: float) -> str:
    """«San Salvador, San Miguel, Santa Tecla y Santa Ana reúnen el 31% del total.» con los
    distritos de mayor a menor esperado (hasta 4; los que haya si son menos). Vacía con un solo distrito."""
    if len(distritos) < 2:
        return ""
    principales = list(distritos[:NUM_DISTRITOS_TEXTO])
    pct = porcentaje_entero(sum(e for _, e in principales), total)
    verbo = "reúnen"
    return f"{_unir([nombre for nombre, _ in principales])} {verbo} el {pct}% del total."


def _plural_dia(dia: str) -> str:
    minuscula = dia.lower()
    return minuscula if minuscula.endswith("s") else minuscula + "s"


def _unir(nombres: Sequence[str]) -> str:
    if len(nombres) == 1:
        return nombres[0]
    return ", ".join(nombres[:-1]) + " y " + nombres[-1]


def _apertura(inicio: date, fin: date) -> str:
    if inicio == fin:
        return f"El {fecha_larga(inicio)} se esperan"
    if inicio.year == fin.year and inicio.month == fin.month:
        return f"Entre el {inicio.day} y el {fin.day} de {MESES[fin.month - 1]} de {fin.year} se esperan"
    desde = f"{inicio.day} de {MESES[inicio.month - 1]}" + (f" de {inicio.year}" if inicio.year != fin.year else "")
    return f"Entre el {desde} y el {fecha_larga(fin)} se esperan"


def _frase_ebriedad(ebriedad: Mapping[Franja, float]) -> str | None:
    """Ebriedad de noche/madrugada vs. de día, con la composición histórica (promedio simple de
    las franjas filtradas de cada grupo). Omite la parte que no aplica."""
    noche = [f for f in (Franja.NOCHE, Franja.MADRUGADA) if f in ebriedad]
    dia = [f for f in (Franja.MANANA, Franja.TARDE) if f in ebriedad]

    def promedio(franjas: list[Franja]) -> int:
        return int(sum(ebriedad[f] for f in franjas) / len(franjas) * 100 + 0.5)

    if noche and dia:
        etiqueta_dia = "del día" if len(dia) == 2 else f"de la {_SUSTANTIVO_FRANJA[dia[0]]}"
        return (
            f"{_de(noche)}, la ebriedad explica cerca del {promedio(noche)}% de los siniestros "
            f"esperados, frente al {promedio(dia)}% {etiqueta_dia}."
        )
    if noche:
        return f"{_de(noche)}, la ebriedad explica cerca del {promedio(noche)}% de los siniestros esperados."
    if dia:
        inicio = "De día" if len(dia) == 2 else f"Por la {_SUSTANTIVO_FRANJA[dia[0]]}"
        return f"{inicio}, la ebriedad explica cerca del {promedio(dia)}% de los siniestros esperados."
    return None


def _de(franjas: list[Franja]) -> str:
    return "De " + " y de ".join(_SUSTANTIVO_FRANJA[f] for f in franjas)


def redactar_resumen(
    *,
    fecha_inicio: date,
    fecha_fin: date,
    total: float,
    ic90: tuple[int, int],
    distritos: Sequence[tuple[str, float]],
    por_franja: Sequence[tuple[Franja, float]],
    por_dia: Sequence[FilaDiaSemana],
    ebriedad_por_franja: Mapping[Franja, float],
) -> ResumenInforme:
    """`distritos`: (nombre, esperado) de los distritos del informe, de mayor a menor.
    `por_franja`: solo las franjas consideradas (las del filtro). `por_dia`: las 7 filas
    (`veces` = 0 si el filtro excluye el día). `ebriedad_por_franja`: proporción histórica de
    la ebriedad en cada franja considerada."""
    franja_top, esperado_franja = max(por_franja, key=lambda x: x[1])
    pct_franja = porcentaje_entero(esperado_franja, total)
    dias_activos = [d for d in por_dia if d.veces > 0]
    # Desempate determinista por orden de la semana (nunca por número de apariciones).
    dia_top = max(dias_activos, key=lambda d: round(d.esperado, 9))

    # --- párrafo
    n = len(distritos)
    if n == 1:
        donde = f"en el distrito {distritos[0][0]}"
    elif n == _TOTAL_DISTRITOS:
        donde = "en el país"
    else:
        donde = f"en los {n} distritos"
    inf, sup = ic90
    if int(total + 0.5) == 1:
        cantidad = "se espera cerca de 1 siniestro"
        apertura = _apertura(fecha_inicio, fecha_fin).removesuffix(" se esperan")
        primera = f"{apertura} {cantidad} {donde}"
    else:
        primera = f"{_apertura(fecha_inicio, fecha_fin)} unos {conteo(total)} siniestros {donde}"
    frases = [f"{primera} (entre {conteo(inf)} y {conteo(sup)} con 90% de probabilidad)."]

    partes = []
    if len(por_franja) > 1:
        partes.append(f"La {_SUSTANTIVO_FRANJA[franja_top]} concentra el {pct_franja}%")
    if len(dias_activos) > 1:
        partes.append(
            f"los {_plural_dia(dia_top.dia)} son, en promedio, los días de mayor riesgo"
        )
    if len(partes) == 2:
        frases.append(f"{partes[0]} y {partes[1]}.")
    elif partes and len(por_franja) > 1:
        frases.append(f"{partes[0]} del total.")
    elif partes:
        frases.append(partes[0][0].upper() + partes[0][1:] + ".")

    if n > 1:
        principales = distritos[:NUM_DISTRITOS_TEXTO]
        pct = porcentaje_entero(sum(e for _, e in principales), total)
        frases.append(f"{_unir([nombre for nombre, _ in principales])} reúnen el {pct}% del total.")

    ebria = _frase_ebriedad(ebriedad_por_franja)
    if ebria:
        frases.append(ebria)

    # --- tabla de 4 filas
    por_dia_txt = dia_top.esperado
    por_dia_fmt = conteo(por_dia_txt) if por_dia_txt >= 10 else f"{por_dia_txt:.1f}"
    nombre_top, esperado_top = distritos[0]
    filas = (
        FilaResumen("Total esperado", f"{conteo(total)} (entre {conteo(inf)} y {conteo(sup)})"),
        FilaResumen("Franja de mayor riesgo", f"{_ETIQUETA_FRANJA[franja_top]}, {pct_franja}% del total"),
        FilaResumen("Día de mayor riesgo (promedio)", f"{dia_top.dia}, {por_dia_fmt} por día"),
        FilaResumen("Distrito con más siniestros esperados", f"{nombre_top}, {esperado_top:,.1f}"),
    )
    return ResumenInforme(texto=" ".join(frases), filas=filas)
