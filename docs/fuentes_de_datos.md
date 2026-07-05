# Fuentes de Datos

Proyecto: **Análisis de patrones espacio-temporales de accidentes de tránsito en El Salvador**
Documento de referencia de procedencia de datos (Etapa 1).

Este documento describe el **origen, tamaño, variables clave, licencia y
limitaciones** de cada fuente. El análisis descriptivo y los hallazgos viven en
los notebooks (NB01–NB02); aquí solo se documenta *de dónde viene* cada dato y
*bajo qué condiciones* puede usarse.

---

## Nota metodológica esencial: naturaleza de los datos de siniestralidad

El microdato oficial a nivel de incidente (con geolocalización, hora y
circunstancias) **no es de acceso público** en El Salvador. El ONASEVI/FONAT
publica únicamente **agregados** (totales anuales por departamento, tipo de
víctima, etc.). Esta limitación estaba prevista como el riesgo principal del
anteproyecto ("FONAT no proporciona datos en bruto, solo resúmenes agregados").

Como respuesta metodológica, los archivos `siniestros.csv` y `victimas.csv`
son un **conjunto sintético calibrado a los agregados oficiales publicados**:
un modelo generativo reparte los totales oficiales sobre el espacio (distritos),
el tiempo (fecha/hora/franja) y las circunstancias, preservando las
distribuciones documentadas (estacionalidad de lluvias, feriados, picos de fin
de semana, composición de víctimas por tipo de usuario).

**Encuadre del alcance (importante para la defensa):** el objetivo del proyecto
es demostrar que la *tubería analítica* (EDA → clustering espacio-temporal →
modelos de conteo → interpretabilidad) **recupera correctamente la estructura
conocida** de los datos. No se presenta como generador de hallazgos
epidemiológicos nuevos. Toda conclusión se interpreta a la luz de que los datos
de siniestralidad son sintéticos-calibrados, no observaciones directas.

Las demás fuentes (censo, clima, catálogo de estaciones, feriados) son **datos
reales oficiales o públicos**, sin síntesis.

---

## Resumen de fuentes

| Archivo                              | Contenido                             | Naturaleza                                    | Período         | Licencia / Términos                             |
| ------------------------------------ | ------------------------------------- | --------------------------------------------- | --------------- | ----------------------------------------------- |
| `siniestros.csv`                     | 89 946 accidentes                     | Sintético calibrado a agregados ONASEVI/FONAT | 2022 – jun 2026 | Uso académico (derivado de agregados públicos)  |
| `victimas.csv`                       | 60 402 víctimas                       | Sintético calibrado (enlazado a siniestros)   | 2022 – jun 2026 | Uso académico                                   |
| `TAB_POB_AREA_1.xlsx`                | Población por distrito (262)          | **Real — oficial**                            | Censo 2024      | Dato oficial público, BCR/ONEC (atribución)     |
| `Estaciones SLV.xlsx`                | Catálogo de estaciones meteorológicas | **Real**                                      | —               | Metadatos Meteostat / WMO                       |
| `clima/` (6 estaciones)              | Clima horario                         | **Real**                                      | 2022–2026       | Meteostat: lib MIT · datos CC BY / CC BY-NC 4.0 |
| `feriados_elsalvador_2022_2026.xlsx` | Calendario de feriados                | **Real**                                      | 2022–2026       | Referencia pública                              |

---

## 1. Siniestros — `siniestros.csv`

- **Contenido:** 89 946 registros de accidentes de tránsito.
- **Naturaleza:** sintético calibrado (ver nota metodológica arriba).
- **Calibración (verificada):** los totales anuales del dataset **coinciden
  exactamente** con las cifras oficiales del Observatorio Nacional de Seguridad
  Vial (FONAT) para 2022–2025:

  | Año  | Siniestros | Lesionados | Fallecidos | Estado                   |
  | ---- | ---------- | ---------- | ---------- | ------------------------ |
  | 2022 | 17 408     | 10 385     | 1 352      | completo                 |
  | 2023 | 18 463     | 11 015     | 1 256      | completo                 |
  | 2024 | 20 301     | 11 954     | 1 303      | completo                 |
  | 2025 | 22 265     | 13 285     | 1 238      | completo                 |
  | 2026 | 11 509     | 7 880      | 734        | **parcial (ene–30 jun)** |

  El año 2026 cubre solo el primer semestre y viene marcado en el campo
  `registro_parcial = "Si"` (los demás años: `"No"`). En los análisis anuales
  se trata por separado para no compararlo contra años completos.
- **Estructura espacial:** cada accidente se asigna a un **distrito** (reforma
  territorial 2023: los antiguos municipios son hoy distritos), con su
  `municipio_2024`, `departamento` y coordenadas GPS (`latitud`, `longitud`).
- **Estructura temporal:** `anio`, `mes`, `dia`, `fecha`, `dia_semana`, `hora`
  y `franja_horaria`. El cargador agrega `fecha_hora` (timestamp completo).
- **Distribución preservada (a validar en EDA):** picos de viernes/sábado,
  aumento en temporada lluviosa (may–oct), efecto de feriados, composición de
  víctimas fatales (motociclistas ~42 %, peatones ~35 %), sobredispersión en
  los conteos (justifica Binomial Negativa sobre Poisson).
- **Fuente de calibración:** Observatorio Nacional de Seguridad Vial, FONAT.
  https://observatoriovial.fonat.gob.sv

## 2. Víctimas — `victimas.csv`

- **Contenido:** 60 402 personas, enlazadas a `siniestros.csv` por
  `id_siniestro`.
- **Naturaleza:** sintético calibrado, consistente con los agregados de
  lesionados y fallecidos por año.
- **Variables clave:** `condicion` (fallecido/lesionado), `tipo_usuario`
  (conductor, pasajero, peatón, motociclista…), `rango_etario`,
  `tipo_siniestro`, `tipo_vehiculo`.

## 3. Censo de población — `TAB_POB_AREA_1.xlsx`

- **Contenido:** población por área (urbano/rural) desglosada por
  departamento → municipio → distrito.
- **Naturaleza:** **dato real oficial.** Archivo crudo del **VII Censo de
  Población y VI de Vivienda 2024**.
- **Fuente:** Banco Central de Reserva / Oficina Nacional de Estadística y
  Censos (BCR/ONEC). https://onec.bcr.gob.sv
- **Uso en el proyecto:** referencia de **exposición**. Se limpia en
  `config.load_censo()` al nivel distrito (262 distritos con población total,
  urbana, rural y % urbano) para calcular tasas per cápita en NB03.
- **Cuadre de control:** la suma de distritos (5 922 921) más la fila
  "No Especificado" (107 055) reconcilia con el total nacional oficial
  **6 029 976** habitantes.
- **Nota de integración:** los nombres de distrito se normalizan (se elimina el
  prefijo numérico). La coincidencia exacta con los nombres de `siniestros.csv`
  se valida en NB03 (clave del join per cápita).

## 4. Catálogo de estaciones — `Estaciones SLV.xlsx`

- **Contenido:** metadatos de las estaciones meteorológicas: `id` (WMO),
  `nombre`, `region`, `latitud`, `longitud`, `elevacion`, `timezone`.
- **Uso:** provee las coordenadas GPS de cada estación para el join
  "estación más cercana" a cada distrito (NB03).
- **Estaciones relevantes:**

  | Carpeta clima | id WMO | Nombre                      | Región                  |
  | ------------- | ------ | --------------------------- | ----------------------- |
  | `IL`          | 78663  | San Salvador / Ilopango     | SS (AMSS)               |
  | `SA`          | 78655  | Santa Ana / El Palmar       | SA                      |
  | `SO`          | 78650  | Acajutla                    | SO                      |
  | `PAZ`         | 78666  | Aeropuerto Intl. / Comalapa | PZ                      |
  | `SM`          | 78670  | San Miguel / El Papalón     | SM                      |
  | `SS`          | 78662  | San Salvador (centro)       | SS                      |
  | —             | 78672  | La Unión                    | UN (sin datos de clima) |

## 5. Clima horario — `clima/`

- **Contenido:** registros horarios de temperatura, humedad, **precipitación
  (`prcp`)**, viento, presión y condición meteorológica.
- **Naturaleza:** **dato real**, obtenido de **Meteostat**, que agrega
  observaciones de servicios meteorológicos nacionales (NOAA, DWD, etc.).
- **Licencia:** la librería Meteostat es **MIT**; los **datos** se distribuyen
  bajo **Creative Commons Attribution 4.0 (CC BY 4.0)** — con la variante
  **CC BY-NC 4.0** (no comercial) según sus Términos. El uso académico de este
  proyecto queda cubierto en ambos casos mediante atribución.
  https://dev.meteostat.net/terms
- **Dos formatos internos (unificados en `config._leer_estacion`):**
  - `IL`, `SA`, `SO`: un archivo por estación (esquema con `date` + `datetime`).
  - `PAZ`, `SM`, `SS`: un archivo por año (esquema `year/month/day/hour` con
    columnas `_source`).
  El cargador consolida ambos en una columna `time` común.
- **Rango temporal:** las estaciones de archivo único (IL/SA/SO) llegan hasta
  feb-2026; las de archivo por año (PAZ/SM/SS) hasta jul-2026. Ambos cubren con
  holgura la ventana de siniestros (2022 – jun 2026).
- **Advertencia de calidad (relevante para el EDA):** por defecto Meteostat
  **sustituye observaciones faltantes con datos de modelo/pronóstico**. Las
  columnas `_source` indican la procedencia de cada valor (`metar`, `isd_lite`
  = observación; `dwd_mosmix`, `metno_forecast` = modelo). La completitud real
  de `prcp` por estación y la decisión sobre qué estación usar se documentan en
  NB02.

## 6. Feriados — `feriados_elsalvador_2022_2026.xlsx`

- **Contenido:** calendario de días feriados de El Salvador, 2022–2026.
- **Uso:** derivar variables de calendario (feriado / víspera de feriado) en
  NB03, para capturar el efecto de feriados sobre la siniestralidad.

---

## Limitaciones documentadas

1. **Datos de siniestralidad sintéticos-calibrados.** No son observaciones
   directas; reproducen la estructura de los agregados oficiales. Las
   conclusiones se interpretan bajo este encuadre. *(Riesgo principal del
   anteproyecto, con esta mitigación.)*

2. **Año 2026 parcial.** Cubre solo enero–30 junio 2026 (marcado con
   `registro_parcial = "Si"`). Se trata por separado en los análisis anuales.

3. **Cobertura climática dispersa.** Solo hay estaciones utilizables en 5 zonas
   (Ilopango/AMSS, Santa Ana, Acajutla, Comalapa, San Miguel). El clima se
   asigna a cada distrito por **proximidad geográfica** a la estación más
   cercana, no por medición local.

4. **Oriente lejano sin estación propia.** La estación de **La Unión (78672)**
   existe en el catálogo pero **no tiene datos de clima** (carpeta vacía). El
   extremo oriental queda cubierto de forma aproximada por San Miguel.

5. **Dos estaciones en San Salvador.** El catálogo tiene `78663`
   (Ilopango, AMSS) y `78662` (San Salvador centro). La segunda tiene muy baja
   completitud de precipitación; la selección entre ambas se justifica en NB02.

6. **Sustitución con datos de modelo.** Parte de los registros de clima
   provienen de modelo/pronóstico (no observación directa), según las columnas
   `_source`.

7. **Parque vehicular / commuting inter-departamental.** El flujo real de
   movilidad hacia San Salvador y La Libertad no se captura con el registro
   vehicular por departamento (limitante prevista en el anteproyecto).

> **Fuera del alcance de Etapa 1:** la infraestructura vial de OpenStreetMap
> (OSMnx) está prevista para la fase espacial de Etapa 2 y no se documenta aquí.

---

## Atribución de fuentes

- **Observatorio Nacional de Seguridad Vial — FONAT.** Estadísticas de
  siniestralidad vial 2022–2025. https://observatoriovial.fonat.gob.sv
- **BCR / ONEC.** VII Censo de Población y VI de Vivienda 2024.
  https://onec.bcr.gob.sv
- **Meteostat.** Datos históricos de clima (CC BY / CC BY-NC 4.0; librería MIT).
  https://meteostat.net