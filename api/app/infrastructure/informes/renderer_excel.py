"""Renderer Excel del informe (openpyxl): una hoja por tabla del contrato JSON más
«Supuestos». Mismas cifras que el JSON (se generan del mismo objeto de dominio)."""

from __future__ import annotations

from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.domain.informe import DIAS_SEMANA, InformePeriodo
from app.domain.models import Franja
from app.infrastructure.informes.textos import FUENTE_DATOS, supuestos_texto

_AZUL = "12355B"
_FMT_ESPERADO = "#,##0.00"
_FMT_ENTERO = "#,##0"
_FMT_PORCENTAJE = '0.00"%"'
_FMT_PROPORCION = "0.00%"


class RendererExcel:
    def renderizar(self, i: InformePeriodo) -> bytes:
        libro = Workbook()
        libro.remove(libro.active)

        self._hoja_resumen(libro, i)

        self._hoja(libro, "Total", ["Concepto", "Valor"], [
            ["Esperado", i.total.esperado],
            ["Rango 90% inferior", i.total.ic90_inf],
            ["Rango 90% superior", i.total.ic90_sup],
        ], formatos={2: _FMT_ESPERADO})
        # Valores enteros de la hoja Total: los dos últimos son conteos
        hoja_total = libro["Total"]
        for fila in (3, 4):
            hoja_total.cell(fila, 2).number_format = _FMT_ENTERO

        self._hoja(libro, "Por distrito", [
            "Código", "Distrito", "Departamento", "Esperado", "Rango 90% inferior",
            "Rango 90% superior", "Tercil", "Tasa por 10 mil hab.",
        ], [
            [d.codigo, d.distrito, d.departamento, d.esperado, d.ic90_inf, d.ic90_sup, d.tercil.value, d.tasa_10mil_hab]
            for d in i.por_distrito
        ], formatos={4: _FMT_ESPERADO, 5: _FMT_ENTERO, 6: _FMT_ENTERO, 8: _FMT_ESPERADO})

        self._hoja(libro, "Por franja", ["Franja", "Esperado"],
                   [[f.franja.value, f.esperado] for f in i.por_franja], formatos={2: _FMT_ESPERADO})

        self._hoja(libro, "Por día de la semana", ["Día", "Esperado por día (promedio)", "Veces en el periodo", "Suma en el periodo"],
                   [[d.dia, d.esperado, d.veces, d.suma] for d in i.por_dia_semana],
                   formatos={2: _FMT_ESPERADO, 3: _FMT_ENTERO, 4: _FMT_ESPERADO})

        franjas = list(Franja)
        self._hoja(libro, "Distrito x franja", ["Código", "Distrito", *[f.value for f in franjas]], [
            [d.codigo, d.distrito, *[d.franjas[f] for f in franjas]] for d in i.distrito_x_franja
        ], formatos={3: _FMT_ESPERADO, 4: _FMT_ESPERADO, 5: _FMT_ESPERADO, 6: _FMT_ESPERADO})

        self._hoja(libro, "Por causa", [
            "Causa", "Esperado", "Rango 90% inferior", "Rango 90% superior", "Porcentaje",
        ], [[c.causa, c.esperado, c.ic90_inf, c.ic90_sup, c.porcentaje] for c in i.por_causa],
            formatos={2: _FMT_ESPERADO, 3: _FMT_ENTERO, 4: _FMT_ENTERO, 5: _FMT_PORCENTAJE},
            colores_causas=(i.colores_causas, 1))

        self._hoja(libro, "Causas por franja", ["Franja", "Causa", "Proporción histórica (2022-2024)"], [
            [x.franja.value, c.causa, c.proporcion] for x in i.causas_por_franja for c in x.causas
        ], formatos={3: _FMT_PROPORCION}, colores_causas=(i.colores_causas, 2))

        self._hoja(libro, "Prioridades", ["#", "Distrito", "Franja", "Día de la semana", "Esperado por día (promedio)", "Veces en el periodo"], [
            [n, p.distrito, p.franja.value, p.dia_semana, p.esperado, p.veces] for n, p in enumerate(i.prioridades, 1)
        ], formatos={5: _FMT_ESPERADO, 6: _FMT_ENTERO})

        self._hoja_supuestos(libro, i)

        salida = BytesIO()
        libro.save(salida)
        return salida.getvalue()

    # ------------------------------------------------------------------ utilidades

    @staticmethod
    def _hoja(libro: Workbook, nombre: str, encabezados: list[str], filas: list[list], *,
              formatos: dict[int, str] | None = None, colores_causas: tuple[dict[str, str], int] | None = None) -> None:
        hoja = libro.create_sheet(nombre)
        hoja.append(encabezados)
        for fila in filas:
            hoja.append(fila)
        _estilo_encabezado(hoja, len(encabezados))
        for col, formato in (formatos or {}).items():
            for celda in hoja.iter_cols(min_col=col, max_col=col, min_row=2):
                for c in celda:
                    c.number_format = formato
        if colores_causas is not None:
            colores, col = colores_causas
            for c in hoja.iter_cols(min_col=col, max_col=col, min_row=2):
                for celda in c:
                    celda.fill = PatternFill("solid", start_color=colores[celda.value].lstrip("#").upper())
                    celda.font = Font(color=_color_texto(colores[celda.value]))
        _ajustar_anchos(hoja)
        hoja.freeze_panes = "A2"

    @staticmethod
    def _hoja_resumen(libro: Workbook, i: InformePeriodo) -> None:
        hoja = libro.create_sheet("Resumen")
        hoja.append(["Resumen"])
        hoja.append([i.resumen.texto])
        hoja.append([])
        hoja.append(["Concepto", "Valor"])
        for fila in i.resumen.filas:
            hoja.append([fila.etiqueta, fila.valor])
        hoja.append([])
        hoja.append([FUENTE_DATOS])
        hoja.column_dimensions["A"].width = 44
        hoja.column_dimensions["B"].width = 52
        hoja.merge_cells("A2:B2")
        hoja["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        hoja.row_dimensions[2].height = 120
        hoja["A1"].font = Font(bold=True, size=14, color=_AZUL)
        for col in (1, 2):
            celda = hoja.cell(4, col)
            celda.font = Font(bold=True, color="FFFFFF")
            celda.fill = PatternFill("solid", start_color=_AZUL)
        hoja[f"A{hoja.max_row}"].font = Font(italic=True, color="5F6B7A")
        hoja.freeze_panes = "A2"

    @staticmethod
    def _hoja_supuestos(libro: Workbook, i: InformePeriodo) -> None:
        p, f = i.parametros, i.parametros.filtros
        hoja = libro.create_sheet("Supuestos")
        filas: list[tuple[str, object]] = [
            ("Parámetros", None),
            ("Fecha de inicio", p.fecha_inicio.isoformat()),
            ("Fecha de fin", p.fecha_fin.isoformat()),
            ("Departamento", f.departamento or "Todos"),
            ("Distritos", ", ".join(f.distritos) or "Todos"),
            ("Franjas", ", ".join(x.value for x in f.franjas) or "Todas"),
            ("Días de la semana", ", ".join(DIAS_SEMANA[d] for d in sorted(f.dias_semana)) or "Todos"),
            ("Generado", i.generado.isoformat(timespec="seconds")),
            (None, None),
            ("Supuestos", None),
            *[(f"• {texto}", None) for texto in supuestos_texto(i)],
            (None, None),
            ("Composición de tipo de vehículo (2022-2024)", "Porcentaje"),
            *[(v.tipo, v.porcentaje) for v in i.supuestos.composicion_vehiculos],
            (None, None),
            ("Limitaciones", None),
            *[(f"{n}. {texto}", None) for n, texto in enumerate(i.limitaciones, 1)],
        ]
        for fila in filas:
            hoja.append(list(fila))
        negritas = {"Parámetros", "Supuestos", "Limitaciones", "Composición de tipo de vehículo (2022-2024)"}
        for fila in hoja.iter_rows():
            if fila[0].value in negritas:
                for celda in fila:
                    if celda.value is not None:
                        celda.font = Font(bold=True, color="FFFFFF")
                        celda.fill = PatternFill("solid", start_color=_AZUL)
            elif isinstance(fila[1].value, float):
                fila[1].number_format = '0.0"%"'
        hoja.column_dimensions["A"].width = 110
        hoja.column_dimensions["B"].width = 24
        for fila in hoja.iter_rows():
            fila[0].alignment = Alignment(wrap_text=True, vertical="top")
        hoja.freeze_panes = "A2"


def _color_texto(hexcolor: str) -> str:
    r, g, b = (int(hexcolor.lstrip("#")[k : k + 2], 16) for k in (0, 2, 4))
    return "FFFFFF" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.6 else "1F2933"


def _estilo_encabezado(hoja, n_columnas: int) -> None:
    for col in range(1, n_columnas + 1):
        celda = hoja.cell(1, col)
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", start_color=_AZUL)
        celda.alignment = Alignment(vertical="center", wrap_text=True)


def _ajustar_anchos(hoja) -> None:
    for col in range(1, hoja.max_column + 1):
        mayor = max((len(str(c.value)) for c in hoja[get_column_letter(col)] if c.value is not None), default=8)
        hoja.column_dimensions[get_column_letter(col)].width = min(max(mayor + 3, 12), 48)
