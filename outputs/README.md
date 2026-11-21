
# Resultados de los once notebooks oficiales

Esta carpeta conserva las métricas, configuraciones, tablas, figuras y registros históricos generados durante el desarrollo del proyecto de predicción de incumplimiento de préstamos de Lending Club.

Los archivos permiten consultar los resultados de los once notebooks oficiales sin necesidad de volver a entrenar los modelos. También sirven como evidencia de los experimentos descritos en el [informe final](../reports/informe_final.md).

Los resultados corresponden a distintas etapas de investigación. Debe distinguirse entre los experimentos preliminares realizados sobre muestras y las evaluaciones posteriores sobre el conjunto de desarrollo completo.

## Inventario de resultados

La siguiente tabla presenta las carpetas y los archivos actualmente publicados en el repositorio.

| Carpeta | Notebook | Productor | Contenido | Archivos |
| --- | ---: | --- | --- | ---: |
| `exploracion_inicial/` | 01 | Notebook 01 | Auditoría inicial, faltantes, imputación, correlaciones, diagnósticos y salidas históricas de consulta | 26 |
| `gradient_boosting_220000/` | 03 | Notebook 03 y `src.boosting` | Búsqueda y resultados históricos de Gradient Boosting sobre la muestra inicial | 4 |
| `exploracion_completa/` | 04 | Notebook 04 | Auditoría de cobertura, tipos de datos, particiones y consultas descriptivas | 6 |
| `benchmark_memoria/` | 04 | Mediciones históricas de memoria | Registros de consumo de memoria y tiempo por método de carga | 4 |
| `comparacion_carga/` | 04 | `src.auditoria_memoria` y `src.benchmark_worker` | Mediciones de lectura directa y por bloques | 6 |
| `svm_muestra/` | 05 | Notebook 05 y `src.svm_muestra` | Resultados iniciales de SVM, configuraciones, particiones y verificación de la muestra | 13 |
| `entrenamiento_26_variables/` | 06 | `src.entrenamiento_inicial` | Baselines y métricas de validación temporal con 26 características | 2 |
| `hiperparametros_randomized_26_variables/` | 06 | Notebook 06 | Primera búsqueda aleatoria y comparación con configuraciones de referencia | 5 |
| `comparacion_variables_temporal/` | 07 | `src.comparacion_variables` | Comparación de conjuntos de 26 y 59 variables y selección temporal dinámica | 5 |
| `comparacion_imputacion/` | 07 | `src.comparacion_imputacion` | Comparación entre mediana, indicadores de ausencia y tratamiento nativo de faltantes | 6 |
| `comparacion_lineales/` | 07 | `src.comparacion_lineales` | Regresión logística y SVM con distintas estrategias de tratamiento de faltantes | 7 |
| `comparacion_umbrales_faltantes/` | 08 | `src.comparacion_umbrales` | Umbrales de 10 %, 30 %, 50 % y 75 %; selección de variables, métricas y figuras | 12 |
| `analisis_sensibilidad_faltantes/` | 08 | `src.sensibilidad_faltantes` | Disponibilidad histórica, sensibilidad entre umbrales, efectos de imputación y métricas temporales | 18 |
| `optimizacion_final/` | 09 | Notebook 09 | Optimización definitiva, configuraciones seleccionadas y resultados por corte temporal | 10 |
| `calibracion_probabilidades/` | 10 | `src.etapas_finales` y notebook 10 | Métodos de calibración, métricas, configuraciones y figuras | 11 |
| `evaluacion_final/` | 11 | `src.etapas_finales` y notebook 11 | Evaluación histórica del test, métricas calibradas, auditoría e intervalos de confianza | 9 |
| `analisis_errores/` | 11 | `src.analisis_errores` | Métricas globales y análisis de errores por segmentos del conjunto de prueba | 2 |

Los conteos corresponden exclusivamente a los archivos publicados en cada carpeta de la rama `main`.

## Archivos de la raíz

Además de las carpetas anteriores, la raíz de `outputs/` conserva los siguientes archivos:

- `auditoria.json`
- `busqueda_arbol.csv`
- `busqueda_boosting.csv`
- `coeficientes_baseline.csv`
- `decisiones_faltantes_logistica_con_indicadores.csv`
- `decisiones_faltantes_logistica_con_precio_lc.csv`
- `decisiones_faltantes_logistica_sin_indicadores.csv`
- `exclusiones_variables.csv`
- `importancia_arbol.csv`
- `importancia_boosting.csv`
- `importancia_gini_arbol.csv`
- `metrics.json`
- `metrics_arbol.json`
- `metrics_boosting.json`
- `reglas_arbol.txt`

Estos archivos corresponden principalmente a la auditoría inicial, los baselines y los experimentos históricos de árboles de decisión y Gradient Boosting.

Se mantienen para preservar la trazabilidad de los resultados y permitir la consulta de los notebooks correspondientes.

Sus métricas no deben confundirse con las obtenidas en la evaluación final sobre el conjunto de desarrollo completo.

## Resultados definitivos del proyecto

Los principales resultados utilizados en el informe final se encuentran en las siguientes rutas:

| Resultado | Archivo |
| --- | --- |
| Auditoría general del dataset | `auditoria.json` |
| Baselines con 26 características | `entrenamiento_26_variables/resumen_modelos_validacion.csv` |
| Comparación temporal de características | `comparacion_variables_temporal/resumen_promedios_y_variabilidad.csv` |
| Comparación de umbrales de faltantes | `comparacion_umbrales_faltantes/resumen_promedios_y_variabilidad.csv` |
| Selección y optimización final | `optimizacion_final/resumen_todas_configuraciones.csv` |
| Parámetros seleccionados | `optimizacion_final/mejores_parametros_por_modelo.csv` |
| Comparación de métodos de calibración | `calibracion_probabilidades/metricas_promedio.csv` |
| Métricas finales del test | `evaluacion_final/metricas_finales_test.csv` |
| Intervalos de confianza del test | `evaluacion_final/intervalos_bootstrap_95.csv` |
| Análisis de errores por segmentos | `analisis_errores/metricas_por_segmento.csv` |

La evaluación histórica definitiva utiliza los préstamos elegibles de 2017–2018, con un total de 225,611 observaciones y los desenlaces `Fully Paid` y `Charged Off`.

Las métricas corresponden a los tres modelos finales calibrados: regresión logística, SVM lineal e HistGradientBoosting.

Las tablas y configuraciones publicadas permiten contrastar los resultados presentados en el informe sin repetir los experimentos completos.

## Alcance de la evidencia conservada

Los archivos publicados documentan experimentos académicos de clasificación crediticia y permiten revisar sus resultados históricos.

No todos corresponden al modelo definitivo. Los experimentos iniciales sobre muestras, las comparaciones de características, la optimización, la calibración y la evaluación final representan etapas metodológicas distintas.

Se conservan sus resultados para facilitar la trazabilidad del trabajo, incluso cuando determinadas figuras preliminares no se incluyen en el informe final.

La existencia de métricas y configuraciones guardadas no equivale a una nueva ejecución independiente ni garantiza que los resultados se reproduzcan en cualquier entorno.

La evaluación final sigue siendo retrospectiva y está sujeta a las limitaciones metodológicas documentadas en el informe.