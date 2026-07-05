# Diccionario de Datos

Proyecto: **Análisis de patrones espacio-temporales de accidentes de tránsito en El Salvador** (Etapa 1).

Describe cada columna de los archivos de datos: tipo, descripción y valores
posibles. Los archivos `siniestros.csv` y `victimas.csv` usan **`;`** como
separador. La procedencia y las licencias están en `fuentes_de_datos.md`.

---

## 1. `siniestros.csv`

Nivel de grano: **un registro por accidente**. 89 946 filas, 22 columnas
originales (+ `fecha_hora`, derivada por el cargador).

| Columna            | Tipo         | Descripción                                                                                                     | Valores / Formato                                                                       |
| ------------------ | ------------ | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `id_siniestro`     | texto        | Identificador único del accidente. Clave de enlace con `victimas.csv`.                                          | `SV-AAAA-NNNNNN` (ej. `SV-2022-000001`)                                                 |
| `anio`             | entero       | Año del accidente.                                                                                              | 2022–2026                                                                               |
| `mes`              | entero       | Mes del accidente.                                                                                              | 1–12                                                                                    |
| `dia`              | entero       | Día del mes.                                                                                                    | 1–31                                                                                    |
| `fecha`            | fecha        | Fecha del accidente.                                                                                            | `AAAA-MM-DD`                                                                            |
| `dia_semana`       | categórico   | Día de la semana.                                                                                               | Lunes, Martes, Miercoles, Jueves, Viernes, Sabado, Domingo                              |
| `hora`             | texto (hora) | Hora del accidente.                                                                                             | `HH:MM` (24 h)                                                                          |
| `franja_horaria`   | categórico   | Franja del día.                                                                                                 | Madrugada (00-05), Manana (06-11), Tarde (12-17), Noche (18-23)                         |
| `latitud`          | decimal      | Latitud del punto del accidente.                                                                                | grados decimales (~13.1–14.5)                                                           |
| `longitud`         | decimal      | Longitud del punto del accidente.                                                                               | grados decimales (~-90.1 a -87.7)                                                       |
| `departamento`     | categórico   | Departamento (14 del país).                                                                                     | ej. San Salvador, Ahuachapán…                                                           |
| `distrito`         | categórico   | Distrito (unidad territorial post-reforma 2023; antiguo "municipio"). Clave del join con censo.                 | 262 posibles                                                                            |
| `municipio_2024`   | categórico   | Municipio según reforma territorial 2024 (agrupa distritos).                                                    | ej. San Salvador Centro                                                                 |
| `zona`             | categórico   | Tipo de zona.                                                                                                   | Urbana, Rural                                                                           |
| `tipo_via`         | categórico   | Tipo de vía donde ocurre el accidente.                                                                          | Carretera nacional, Via departamental, Calle urbana, Calle rural                        |
| `tipo_siniestro`   | categórico   | Tipo de accidente.                                                                                              | Colision, Atropello, Choque contra objeto fijo, Volcamiento, Caracteristicas especiales |
| `factor_causante`  | categórico   | Causa principal atribuida.                                                                                      | 17 categorías (ver lista abajo)                                                         |
| `condicion_via`    | categórico   | Estado de la superficie de la vía.                                                                              | Seca, Humeda, Con lodo/derrame                                                          |
| `tipo_vehiculo`    | categórico   | Vehículo principal involucrado en el accidente.                                                                 | Motocicleta, Automovil, Pick-up, Camion, Bus, Otro                                      |
| `fallecidos`       | entero       | N.º de personas fallecidas en el accidente.                                                                     | ≥ 0                                                                                     |
| `lesionados`       | entero       | N.º de personas lesionadas en el accidente.                                                                     | ≥ 0                                                                                     |
| `registro_parcial` | categórico   | Marca si el registro pertenece a un período incompleto. `Si` solo en 2026 (ene–30 jun).                         | No, Si                                                                                  |
| `fecha_hora`       | datetime     | **Derivada** por `config.load_siniestros()`: `fecha` + `hora` como timestamp completo. No está en el CSV crudo. | `AAAA-MM-DD HH:MM:SS`                                                                   |

**Valores de `factor_causante` (17):** Distracción del Conductor, Estado de
Ebriedad o Droga, Invadir Carril, No Guardar Distancia Reglamentaria,
Inexperiencia, No Respetar Señales de Tránsito, Velocidad Excesiva,
Adelantamiento Antireglamentario, Imprudencia del Peatón, Giro Incorrecto, Mal
Estado del Vehículo, Circular en Reversa, Falla Mecánica, Carga Mal
Acondicionada, Deslumbramiento, Enfermedad, Otros.

---

## 2. `victimas.csv`

Nivel de grano: **un registro por persona (víctima)**. 60 402 filas.
Enlaza con `siniestros.csv` mediante `id_siniestro` (relación muchos-a-uno:
un accidente puede tener varias víctimas).

| Columna          | Tipo       | Descripción                                                           | Valores / Formato                                                                       |
| ---------------- | ---------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `id_victima`     | texto      | Identificador único de la víctima.                                    | texto                                                                                   |
| `id_siniestro`   | texto      | Accidente al que pertenece. **Clave foránea** hacia `siniestros.csv`. | `SV-AAAA-NNNNNN`                                                                        |
| `anio`           | entero     | Año (redundante con el del siniestro, para conveniencia).             | 2022–2026                                                                               |
| `fecha`          | fecha      | Fecha del accidente asociado.                                         | `AAAA-MM-DD`                                                                            |
| `condicion`      | categórico | Resultado para la persona.                                            | Lesionado, Fallecido                                                                    |
| `tipo_usuario`   | categórico | Rol de la persona en la vía.                                          | Conductor, Pasajero, Peaton, Ciclista                                                   |
| `rango_etario`   | categórico | Grupo de edad (quinquenal).                                           | 0-15, 16-20, 21-25, …, 66-70, 71+ (13 grupos)                                           |
| `tipo_siniestro` | categórico | Tipo de accidente asociado.                                           | Colision, Atropello, Choque contra objeto fijo, Volcamiento, Caracteristicas especiales |
| `tipo_vehiculo`  | categórico | Vehículo **de la víctima** (distinto del vehículo del siniestro).     | Automovil, Motocicleta, Bicicleta, Ninguno (peaton)                                     |

**Nota sobre `tipo_vehiculo`:** no comparte universo con el `tipo_vehiculo` de
`siniestros.csv`. Aquí describe el modo de transporte *de la persona* (incluye
Bicicleta y "Ninguno (peaton)"); allá describe el vehículo principal *del
accidente*. Son campos con semántica distinta.

---

## 3. `TAB_POB_AREA_1.xlsx` (censo)

Archivo oficial crudo del VII Censo de Población 2024. Tiene **encabezados
jerárquicos en dos filas** y filas de título/subtotales que no son datos. El
cargador `config.load_censo()` lo transforma en una tabla plana a nivel distrito.

### 3.1 Estructura CRUDA del archivo (lo que hay en el `.xlsx`)

- **Filas 0–2:** títulos del censo (no son datos).
- **Fila 3 — encabezado nivel 1:** `Departamento de residencia`,
  `Municipio de residencia`, `Distrito de residencia`, `Total`, `1. Urbano`,
  `2. Rural`.
- **Fila 4 — encabezado nivel 2:** bajo `1. Urbano` y bajo `2. Rural` se
  subdividen en `1. Hombre` y `2. Mujer` (de ahí las 8 columnas).
- **Fila 5:** vacía.
- **Fila 6 en adelante:** datos. Incluye filas de **subtotal** (donde
  `Municipio` o `Distrito` = `TOTAL`), la fila `No Especificado` y una nota de
  fuente al pie — todas se descartan en la limpieza.

Disposición de las 8 columnas crudas:

| Pos | Encabezado nivel 1         | Encabezado nivel 2 |
| --- | -------------------------- | ------------------ |
| 0   | Departamento de residencia | —                  |
| 1   | Municipio de residencia    | —                  |
| 2   | Distrito de residencia     | —                  |
| 3   | Total                      | —                  |
| 4   | 1. Urbano                  | 1. Hombre          |
| 5   | 1. Urbano                  | 2. Mujer           |
| 6   | 2. Rural                   | 1. Hombre          |
| 7   | 2. Rural                   | 2. Mujer           |

### 3.2 Salida LIMPIA de `config.load_censo()` (lo que consumen los notebooks)

262 filas (una por distrito). Los conteos de sexo se agregan en totales urbano
y rural; se descartan filas TOTAL/No Especificado; se quitan los prefijos
numéricos de los nombres.

| Columna        | Tipo       | Descripción                                                       |
| -------------- | ---------- | ----------------------------------------------------------------- |
| `departamento` | categórico | Departamento (sin prefijo numérico).                              |
| `municipio`    | categórico | Municipio de residencia (Censo 2024).                             |
| `distrito`     | categórico | Distrito de residencia. Clave del join per cápita con siniestros. |
| `poblacion`    | entero     | Población total del distrito (columna `Total`).                   |
| `urbano`       | entero     | Población urbana (`1. Urbano`: Hombre + Mujer).                   |
| `rural`        | entero     | Población rural (`2. Rural`: Hombre + Mujer).                     |
| `pct_urbano`   | decimal    | Proporción urbana (`urbano / poblacion`), 0–1.                    |

> **Cuadre de control:** suma de distritos (5 922 921) + fila `No Especificado`
> (107 055) = total nacional oficial **6 029 976**.

---

## 4. `Estaciones SLV.xlsx` (catálogo de clima) — `config.load_estaciones()`

| Columna     | Tipo    | Descripción                                        |
| ----------- | ------- | -------------------------------------------------- |
| `id`        | entero  | Identificador WMO de la estación (ej. 78663).      |
| `nombre`    | texto   | Nombre de la estación.                             |
| `pais`      | texto   | Código de país (SV).                               |
| `region`    | texto   | Código de región interna (SS, SA, SO, SM, PZ, UN). |
| `latitud`   | decimal | Latitud de la estación.                            |
| `longitud`  | decimal | Longitud de la estación.                           |
| `elevacion` | entero  | Elevación en metros.                               |
| `timezone`  | texto   | Zona horaria (America/El_Salvador).                |

---

## 5. `clima/` (horario, por estación) — `config.load_clima()` / `load_clima_all()`

El cargador unifica dos formatos en una columna `time` común y conserva las
variables meteorológicas. Columnas presentes según el formato de origen:

| Columna      | Tipo     | Descripción                                                                                              | Presente en     |
| ------------ | -------- | -------------------------------------------------------------------------------------------------------- | --------------- |
| `time`       | datetime | Marca temporal horaria (derivada/unificada por el cargador).                                             | todas           |
| `temp`       | decimal  | Temperatura del aire (°C).                                                                               | todas           |
| `rhum`       | decimal  | Humedad relativa (%).                                                                                    | todas           |
| `prcp`       | decimal  | **Precipitación (mm).** Variable climática principal del proyecto.                                       | todas           |
| `wdir`       | decimal  | Dirección del viento (grados).                                                                           | todas           |
| `wspd`       | decimal  | Velocidad del viento (km/h).                                                                             | todas           |
| `pres`       | decimal  | Presión atmosférica (hPa).                                                                               | todas           |
| `coco`       | entero   | Código de condición meteorológica (Meteostat).                                                           | todas           |
| `dwpt`       | decimal  | Punto de rocío (°C).                                                                                     | IL, SA, SO      |
| `snow`       | decimal  | Nieve (mm) — irrelevante en El Salvador.                                                                 | IL, SA, SO      |
| `wpgt`       | decimal  | Ráfaga máxima de viento (km/h).                                                                          | ambos (parcial) |
| `tsun`       | decimal  | Minutos de sol.                                                                                          | IL, SA, SO      |
| `cldc`       | decimal  | Nubosidad.                                                                                               | PAZ, SM, SS     |
| `*_source`   | texto    | Procedencia de cada variable (`metar`/`isd_lite` = observación; `dwd_mosmix`/`metno_forecast` = modelo). | PAZ, SM, SS     |
| `station_id` | entero   | ID de estación en el archivo crudo.                                                                      | IL, SA, SO      |

> Qué variables de clima se conservan para el modelado se decide en el EDA
> (NB02). Para Etapa 1 la variable de interés central es `prcp`.

---

## 6. `feriados_elsalvador_2022_2026.xlsx` — `config.load_feriados()`

Calendario de feriados de El Salvador (2022–2026). 70 filas. Se usa en NB03 para
derivar variables de calendario (feriado / víspera) sobre la grilla de fechas.

| Columna          | Tipo       | Descripción                                                                                                     | Valores / Formato                 |
| ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `anio`           | entero     | Año del feriado.                                                                                                | 2022–2026                         |
| `fecha`          | fecha      | Fecha del feriado. Clave del join de calendario.                                                                | `AAAA-MM-DD`                      |
| `mes`            | entero     | Mes.                                                                                                            | 1–12                              |
| `dia`            | entero     | Día del mes.                                                                                                    | 1–31                              |
| `dia_semana`     | categórico | Día de la semana.                                                                                               | Lunes … Domingo                   |
| `nombre_feriado` | texto      | Nombre del feriado (ej. "Ano Nuevo", "Viernes Santo").                                                          | texto                             |
| `tipo`           | categórico | Ámbito del feriado.                                                                                             | Nacional, Local (San Salvador), … |
| `es_feriado`     | categórico | Marca de feriado (siempre `Si` en el catálogo; el "No" surge al unir con la grilla completa de fechas en NB03). | Si                                |

> **Nota para NB03:** los feriados de tipo *Local* aplican solo a su distrito
> (p. ej. las Fiestas Agostinas de San Salvador), no a todo el país. El join de
> calendario debe respetar ese alcance para no marcar como feriado un día local
> en distritos donde no lo es.