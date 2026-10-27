"""Adaptador de comparación histórica: observado real (columna `n_siniestros`
de matriz_frecuencia_v2.csv) contra predicho por el modelo de NB07, sumados
por distrito sobre la partición de PRUEBA del histórico (224,952 filas,
2025-2026).

Regla D1 de NB08 (`exigir_particion_de_diagnostico`): el conjunto de prueba
es la única fuente válida para reportar una métrica de desempeño — nunca se
mezcla con train, porque el modelo ya vio esas filas al entrenar y eso
infla la calibración/Spearman de forma artificial. Por eso este adaptador
filtra explícitamente `split == "test"` antes de agregar; usar el CSV
completo (train 2022-2024 incluido) repetiría exactamente el sesgo que esa
regla existe para prevenir en los notebooks — solo que aquí, al ser código
de prototipo y no notebook, no hay ninguna guardia que lo detecte.

Se carga y valida UNA sola vez en el lifespan (ver app/main.py) — correr el
modelo sobre cientos de miles de filas por request sería un desperdicio,
además de arriesgar latencia; se cachea igual que el resto de los
artefactos.

CRÍTICO: este adaptador nunca reimplementa la inferencia. Recibe un
`PredictorFrecuenciaPort` ya construido y validado (`XGBoostFrecuenciaAdapter`
en producción) y le delega `predecir_lote()` — es el único lugar que sabe
sumar `log_tasa_base` al base_margin (ver api/README.md §6). Si este
adaptador construyera su propio DMatrix, repetiría exactamente el fallo
silencioso que ese diseño existe para prevenir."""

from __future__ import annotations

import csv
from pathlib import Path

from app.domain.models import ComparacionDistrito, FeaturesCelda
from app.domain.ports import PredictorFrecuenciaPort

_COLUMNAS_REQUERIDAS = {
    "distrito_key",
    "n_siniestros",
    "es_lluviosa",
    "es_finde",
    "dia_semana",
    "periodo_agostino",
    "es_feriado",
    "prcp_mensual",
    "pct_urbano",
    "poblacion_baja",
    "log_exposicion_v2",
    "split",
}

_PARTICION_DIAGNOSTICO = "test"


class ComparacionHistoricaRepositoryError(RuntimeError):
    """`matriz_frecuencia_v2.csv` no existe o no tiene las columnas esperadas."""


def _fila_a_features(fila: dict[str, str]) -> FeaturesCelda:
    return FeaturesCelda(
        es_lluviosa=int(fila["es_lluviosa"]),
        es_finde=int(fila["es_finde"]),
        dia_semana=int(fila["dia_semana"]),
        periodo_agostino=int(fila["periodo_agostino"]),
        es_feriado=int(fila["es_feriado"]),
        prcp_mensual=float(fila["prcp_mensual"]),
        pct_urbano=float(fila["pct_urbano"]),
        poblacion_baja=int(fila["poblacion_baja"]),
        log_exposicion_v2=float(fila["log_exposicion_v2"]),
    )


class ComparacionHistoricaAdapter:
    def __init__(self, ruta_matriz: Path, predictor: PredictorFrecuenciaPort) -> None:
        self._ruta = ruta_matriz
        self._predictor = predictor
        self._resultado: list[ComparacionDistrito] | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta.exists():
            raise ComparacionHistoricaRepositoryError(f"No se encontró {self._ruta}")

        distritos_en_orden: list[str] = []
        indice_por_distrito: dict[str, int] = {}
        observado_por_distrito: list[float] = []
        filas_por_distrito: list[list[FeaturesCelda]] = []

        with self._ruta.open(encoding="utf-8") as fh:
            lector = csv.DictReader(fh)
            faltantes = _COLUMNAS_REQUERIDAS - set(lector.fieldnames or [])
            if faltantes:
                raise ComparacionHistoricaRepositoryError(
                    f"{self._ruta} no tiene las columnas esperadas: faltan {faltantes}"
                )

            for fila in lector:
                if fila["split"] != _PARTICION_DIAGNOSTICO:
                    continue

                codigo = fila["distrito_key"]
                idx = indice_por_distrito.get(codigo)
                if idx is None:
                    idx = len(distritos_en_orden)
                    indice_por_distrito[codigo] = idx
                    distritos_en_orden.append(codigo)
                    observado_por_distrito.append(0.0)
                    filas_por_distrito.append([])

                observado_por_distrito[idx] += float(fila["n_siniestros"])
                filas_por_distrito[idx].append(_fila_a_features(fila))

        if not distritos_en_orden:
            raise ComparacionHistoricaRepositoryError(f"{self._ruta}: no tiene filas")

        # Una sola llamada en lote sobre TODO el histórico (más eficiente que
        # una llamada por distrito) — se reconstruye la agrupación después
        # con los límites de cada tramo.
        todas_las_filas: list[FeaturesCelda] = [f for grupo in filas_por_distrito for f in grupo]
        todas_las_predicciones = self._predictor.predecir_lote(todas_las_filas)

        resultado: list[ComparacionDistrito] = []
        cursor = 0
        for i, codigo in enumerate(distritos_en_orden):
            n = len(filas_por_distrito[i])
            predicho = sum(todas_las_predicciones[cursor : cursor + n])
            cursor += n
            resultado.append(
                ComparacionDistrito(
                    distrito_codigo=codigo,
                    observado=observado_por_distrito[i],
                    predicho=predicho,
                )
            )

        self._resultado = resultado

    def por_distrito(self) -> list[ComparacionDistrito]:
        if self._resultado is None:
            raise ComparacionHistoricaRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar)."
            )
        return list(self._resultado)
