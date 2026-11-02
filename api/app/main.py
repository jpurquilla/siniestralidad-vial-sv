"""Punto de entrada de la API. El modelo y todos sus adaptadores se cargan y
validan UNA sola vez aquí, en el lifespan — nunca por request. Si cualquier
validación falla, la aplicación se niega a arrancar (ver cada adaptador para
el detalle de qué valida y por qué)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.application.calcular_equidad import CalcularEquidad
from app.application.predecir_celda import DistritoNoEncontradoError, PredecirCelda
from app.application.ranking_distritos import RankingDistritos
from app.config import settings
from app.infrastructure.analitica.comparacion_historica_adapter import (
    ComparacionHistoricaAdapter,
)
from app.infrastructure.analitica.impacto_ley_adapter import ImpactoLeyAdapter
from app.infrastructure.analitica.siniestro_punto_adapter import SiniestroPuntoAdapter
from app.infrastructure.analitica.usuarios_vulnerables_adapter import (
    UsuariosVulnerablesAdapter,
)
from app.infrastructure.geo.distrito_repository import DistritoRepository
from app.infrastructure.geo.geojson_adapter import GeoJsonAdapter
from app.infrastructure.modelo.calendario_adapter import CalendarioAdapter
from app.infrastructure.modelo.clima_adapter import ClimaAdapter
from app.infrastructure.modelo.exposicion_adapter import ExposicionAdapter
from app.domain.models import COLUMNAS_MODELO
from app.infrastructure.modelo.info_modelo import cargar_info_modelo
from app.infrastructure.modelo.parque_adapter import ParqueAdapter
from app.infrastructure.modelo.tercil_repository import TercilRepository
from app.infrastructure.modelo.xgboost_adapter import XGBoostFrecuenciaAdapter
from app.infrastructure.web.routers import (
    comparacion_historica,
    distritos,
    equidad,
    impacto_ley,
    modelo,
    predicciones,
    ranking,
    salud,
    siniestros,
    usuarios_vulnerables,
)

logging.basicConfig(level=settings.nivel_log, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Arrancando %s...", settings.nombre_app)

    xgboost_adapter = XGBoostFrecuenciaAdapter(
        settings.directorio_modelos, settings.ruta_especificacion_nb12
    )
    exposicion_adapter = ExposicionAdapter(settings.ruta_matriz_frecuencia, settings.ruta_especificacion_nb05)
    calendario_adapter = CalendarioAdapter(settings.ruta_feriados)
    parque_adapter = ParqueAdapter(settings.ruta_matriz_frecuencia)
    distrito_repo = DistritoRepository(settings.ruta_matriz_frecuencia, settings.ruta_censo_poblacion)
    clima_adapter = ClimaAdapter(settings.ruta_matriz_frecuencia)
    tercil_repo = TercilRepository(settings.ruta_terciles)
    geojson_adapter = GeoJsonAdapter(settings.ruta_geojson_distritos)
    impacto_ley_adapter = ImpactoLeyAdapter(settings.ruta_resultados_nb06, settings.ruta_parque_vehicular)
    usuarios_vulnerables_adapter = UsuariosVulnerablesAdapter(settings.ruta_matriz_severidad)
    # Reusa xgboost_adapter.predecir_lote() — nunca reimplementa la
    # inferencia (ver api/README.md §6 y el docstring del adaptador).
    comparacion_historica_adapter = ComparacionHistoricaAdapter(
        settings.ruta_matriz_frecuencia, xgboost_adapter
    )
    siniestro_punto_repo = SiniestroPuntoAdapter(
        settings.ruta_siniestros, nombre_distrito=distrito_repo.nombre_para_mostrar
    )

    try:
        xgboost_adapter.cargar_y_validar()
        exposicion_adapter.cargar_y_validar()
        calendario_adapter.cargar_y_validar()
        distrito_repo.cargar_y_validar()
        parque_adapter.cargar_y_validar()
        clima_adapter.cargar_y_validar()
        tercil_repo.cargar_y_validar()
        impacto_ley_adapter.cargar_y_validar()
        usuarios_vulnerables_adapter.cargar_y_validar()
        # Corre el modelo en lote sobre las 676,504 filas históricas UNA
        # sola vez acá — nunca por request (ver docstring del adaptador).
        comparacion_historica_adapter.cargar_y_validar()
        siniestro_punto_repo.cargar_y_validar()
        info_modelo = cargar_info_modelo(
            settings.ruta_metricas_nb12,
            xgboost_adapter.version_modelo,
            xgboost_adapter.fecha_entrenamiento,
            list(COLUMNAS_MODELO),
        )
    except Exception:
        logger.exception(
            "El servicio NO arrancó: falló la validación de un artefacto. "
            "Ver el traceback arriba para el detalle exacto."
        )
        raise

    app.state.predecir_celda_uc = PredecirCelda(
        xgboost_adapter,
        distrito_repo,
        clima_adapter,
        calendario_adapter,
        exposicion_adapter,
        parque_adapter,
    )
    app.state.ranking_uc = RankingDistritos(
        xgboost_adapter,
        distrito_repo,
        clima_adapter,
        calendario_adapter,
        exposicion_adapter,
        parque_adapter,
        tercil_repo,
    )
    app.state.distrito_repo = distrito_repo
    app.state.geojson_adapter = geojson_adapter
    app.state.info_modelo = info_modelo
    app.state.impacto_ley_adapter = impacto_ley_adapter
    app.state.usuarios_vulnerables_adapter = usuarios_vulnerables_adapter
    app.state.comparacion_historica_adapter = comparacion_historica_adapter
    app.state.calcular_equidad_uc = CalcularEquidad(
        comparacion_historica_adapter, distrito_repo, settings.umbral_pct_urbano_equidad
    )
    app.state.siniestro_punto_repo = siniestro_punto_repo
    app.state.estado_salud = {
        "estado": "ok",
        "modelo_cargado": True,
        "version_modelo": xgboost_adapter.version_modelo,
        "log_tasa_base": xgboost_adapter.log_tasa_base,
        "prediccion_canaria_ok": True,
        "version": settings.version_app,
    }

    logger.info("%s arrancó correctamente.", settings.nombre_app)
    yield
    logger.info("Apagando %s.", settings.nombre_app)


app = FastAPI(title=settings.nombre_app, version=settings.version_app, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origenes,
    allow_credentials=False,  # la API no usa cookies ni sesiones
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DistritoNoEncontradoError)
async def _distrito_no_encontrado(request: Request, exc: DistritoNoEncontradoError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


app.include_router(salud.router)
app.include_router(modelo.router)
app.include_router(distritos.router)
app.include_router(predicciones.router)
app.include_router(ranking.router)
app.include_router(impacto_ley.router)
app.include_router(usuarios_vulnerables.router)
app.include_router(comparacion_historica.router)
app.include_router(equidad.router)
app.include_router(siniestros.router)
