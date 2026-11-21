# Resultados de los diez notebooks oficiales

| Carpeta | Notebook | Productor | Contenido | Archivos |
|---|---:|---|---|---:|
| `analisis_errores/` | 11 | `src.analisis_errores` | Métricas globales y por segmento de test | 2 |
| `analisis_sensibilidad_faltantes/` | 08 | `src.sensibilidad_faltantes` | Cobertura, sensibilidad 50/75, métricas, intervalos y figuras | 18 |
| `benchmark_memoria/` | 04 | `Mediciones de memoria por fase ya registradas` | Cuatro registros de memoria y tiempo usados por el anexo | 4 |
| `calibracion_probabilidades/` | 10 | `src.etapas_finales y notebook 10` | Métodos, métricas, variables, probabilidades de validación y figuras | 12 |
| `comparacion_carga/` | 04 | `src.auditoria_memoria / src.benchmark_worker` | Seis mediciones de carga directa y por bloques | 6 |
| `comparacion_imputacion/` | 07 | `src.comparacion_imputacion` | Mediana, indicadores y NaN nativo | 6 |
| `comparacion_lineales/` | 07 | `src.comparacion_lineales` | Logística y SVM con las estrategias de preparación | 7 |
| `comparacion_umbrales_faltantes/` | 08 | `src.comparacion_umbrales` | Umbrales 10/30/50/75, selecciones, métricas y figuras | 12 |
| `comparacion_variables_temporal/` | 07 | `src.comparacion_variables` | 26/59/selección dinámica; listas y resultados | 5 |
| `entrenamiento_26_variables/` | 06 | `src.entrenamiento_inicial` | Referencia inicial con 26 variables | 2 |
| `evaluacion_final/` | 11 | `src.etapas_finales y notebook 11` | Configuración, métricas calibradas, predicciones test e intervalos | 10 |
| `exploracion_completa/` | 04 | `Notebook 04` | Auditoría de faltantes hasta 2012 | 1 |
| `hiperparametros_randomized_26_variables/` | 06 | `Notebook 06` | Primera búsqueda aleatoria y comparación con configuraciones originales | 5 |
| `optimizacion_final/` | 09 | `Notebook 09` | Optimización definitiva; tablas canónicas para retomar la búsqueda | 10 |

Los archivos de la raíz corresponden al baseline, al árbol y al Gradient Boosting inicial. El árbol y Gradient Boosting son experimentos de los notebooks 02 y 03; las referencias históricas se conservan sin cambios. El notebook 02 coordina actualmente la reproducción del árbol.

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

## Resultados finales

Las métricas finales están en `evaluacion_final/metricas_finales_test.csv`: test 2017–2018, 225,611 préstamos con resultado final. `predicciones_test.csv.gz` permite reconstruir errores por segmentos utilizando el CSV original. Se conservan `predicciones_validacion.csv.gz` y las configuraciones para reproducir los gráficos de calibración sin volver a ajustar modelos.
