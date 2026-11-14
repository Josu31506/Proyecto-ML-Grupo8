# Resultados de los diez notebooks oficiales

| Carpeta | Notebook | Productor | Contenido | Archivos |
|---|---:|---|---|---:|
| `analisis_errores/` | 10 | `src.analisis_errores` | Métricas globales y por segmento de test | 2 |
| `analisis_sensibilidad_faltantes/` | 07 | `src.sensibilidad_faltantes` | Cobertura, sensibilidad 50/75, métricas, intervalos y figuras | 18 |
| `benchmark_memoria/` | 04 | `Mediciones de memoria por fase ya registradas` | Cuatro registros de memoria y tiempo usados por el anexo | 4 |
| `calibracion_probabilidades/` | 09 | `src.etapas_finales y notebook 09` | Métodos, métricas, variables, probabilidades de validación y figuras | 12 |
| `comparacion_carga/` | 04 | `src.auditoria_memoria / src.benchmark_worker` | Seis mediciones de carga directa y por bloques | 6 |
| `comparacion_imputacion/` | 06 | `src.comparacion_imputacion` | Mediana, indicadores y NaN nativo | 6 |
| `comparacion_lineales/` | 06 | `src.comparacion_lineales` | Logística y SVM con las estrategias de preparación | 7 |
| `comparacion_umbrales_faltantes/` | 07 | `src.comparacion_umbrales` | Umbrales 10/30/50/75, selecciones, métricas y figuras | 12 |
| `comparacion_variables_temporal/` | 06 | `src.comparacion_variables` | 26/59/selección dinámica; listas y resultados | 5 |
| `entrenamiento_26_variables/` | 05 | `src.entrenamiento_inicial` | Referencia inicial con 26 variables | 2 |
| `evaluacion_final/` | 10 | `src.etapas_finales y notebook 10` | Configuración, métricas calibradas, predicciones test e intervalos | 10 |
| `exploracion_completa/` | 04 | `Notebook 04` | Auditoría de faltantes hasta 2012 | 1 |
| `hiperparametros_randomized_26_variables/` | 05 | `Notebook 05` | Primera búsqueda aleatoria y comparación con configuraciones originales | 5 |
| `optimizacion_final/` | 08 | `Notebook 08` | Optimización definitiva; tablas canónicas para retomar la búsqueda | 10 |

Los archivos de la raíz corresponden al baseline, al árbol y al Gradient Boosting inicial. Sus productores son `src.run_baseline`, `src.run_arbol` y `src.run_boosting`.

## Archivos de la raíz

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

## Uso y reproducción

Las métricas finales están en `evaluacion_final/metricas_finales_test.csv`: test 2017–2018, 225,611 préstamos con resultado final. `predicciones_test.csv.gz` permite reconstruir errores por segmentos utilizando el CSV original. Se conservan `predicciones_validacion.csv.gz` y las configuraciones para reproducir los gráficos de calibración sin volver a ajustar modelos.

El Gradient Boosting inicial usa objetivo de 5,000 préstamos y 5,002 efectivos. Sus resultados no constituyen una comparación equivalente con el baseline de 220 mil.

Las comparaciones de variables e imputación y la sensibilidad se regeneran por separado mediante los controles de los notebooks 06 y 07, utilizando los módulos directos de `src/`. Las ejecuciones nuevas se guardan en `reproducciones/` cuando se activan; no se crea esa carpeta para conservar copias antiguas.

La búsqueda final utiliza directamente sus tablas canónicas de resultados y variables para retomar una ejecución. Las predicciones intermedias de sensibilidad y las tablas parciales se regeneran cuando se ejecuta su experimento; no son dependencias de lectura de los reportes oficiales. Las figuras principales están aquí o en `reports/figures/`.

El anexo de memoria conserva las mediciones ya registradas por fase; la rutina de auditoría actual reproduce la comprobación de tipos y la comparación del tiempo de lectura. No se volvió a medir memoria, entrenar modelos ni ejecutar búsquedas completas durante esta limpieza.

La ejecución pendiente de Boosting con objetivo 220,000 usará `gradient_boosting_220000/` y figuras en `reports/figures/gradient_boosting_220000/`. Allí se escribirán `metrics_boosting.json`, `busqueda_boosting.csv`, `importancia_boosting.csv` y `muestra_verificada.json` cuando el usuario la ejecute. Estas rutas no contienen resultados nuevos todavía. Los JSON y figuras anteriores de 5,002 siguen intactos.
