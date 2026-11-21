# Inventario de figuras del informe final

Este inventario corresponde a las imágenes insertadas en [el informe final](../informe_final.md): **24 figuras principales**, numeradas del 1 al 24, y **dos figuras del Anexo A**, numeradas A.1 y A.2. Los nombres y las rutas de los PNG se conservan aunque sus prefijos numéricos históricos sean distintos de los números del informe.

## Figuras principales

| Figura | Sección | Archivo | Contenido y procedencia |
| --- | --- | --- | --- |
| 1 | 3 | [exploracion/distribucion_objetivo_completo.png](exploracion/distribucion_objetivo_completo.png) | Distribución de los dos desenlaces en toda la población elegible. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb). |
| 2 | 4 | [seleccion_variables/sensibilidad/01_disponibilidad_anual.png](seleccion_variables/sensibilidad/01_disponibilidad_anual.png) | Disponibilidad de variables a través de los años. Fuente: elaboración propia; [notebook 08](../../notebooks/08_umbrales_sensibilidad.ipynb) y [disponibilidad anual](../../outputs/analisis_sensibilidad_faltantes/disponibilidad_anual_variables.csv). |
| 3 | 4 | [seleccion_variables/comparacion_conjuntos.png](seleccion_variables/comparacion_conjuntos.png) | Comparación de conjuntos y de indicadores de faltantes. Fuente: elaboración propia; [notebook 07](../../notebooks/07_comparacion_variables_imputacion.ipynb), [comparación temporal de conjuntos](../../outputs/comparacion_variables_temporal/resumen_promedios_y_variabilidad.csv). |
| 4 | 4 | [seleccion_variables/umbrales/01_roc_auc_promedio.png](seleccion_variables/umbrales/01_roc_auc_promedio.png) | ROC-AUC medio según el límite de faltantes. Fuente: elaboración propia; [notebook 08](../../notebooks/08_umbrales_sensibilidad.ipynb) y [comparación de umbrales](../../outputs/comparacion_umbrales_faltantes/resumen_promedios_y_variabilidad.csv). |
| 5 | 4 | [seleccion_variables/umbrales/03_variables_por_corte.png](seleccion_variables/umbrales/03_variables_por_corte.png) | Número de variables admitidas por corte y umbral. Fuente: elaboración propia; [conteo de variables](../../outputs/comparacion_umbrales_faltantes/conteo_variables_por_corte_y_umbral.csv), [notebook 08](../../notebooks/08_umbrales_sensibilidad.ipynb) y [comparación de umbrales](../../outputs/comparacion_umbrales_faltantes/resumen_promedios_y_variabilidad.csv). |
| 6 | 4 | [seleccion_variables/sensibilidad/03_delta_roc_anual.png](seleccion_variables/sensibilidad/03_delta_roc_anual.png) | Cambio anual de ROC-AUC al pasar de 50% a 75%. Fuente: elaboración propia; [diferencias anuales](../../outputs/analisis_sensibilidad_faltantes/metricas_anuales_50_vs_75.csv), [notebook 08](../../notebooks/08_umbrales_sensibilidad.ipynb) y [comparación de umbrales](../../outputs/comparacion_umbrales_faltantes/resumen_promedios_y_variabilidad.csv). |
| 7 | 4 | [seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_il_acct.png](seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_il_acct.png) | Distribución observada e imputada de la antigüedad de crédito a plazos. Fuente: elaboración propia; [resumen del efecto de imputación](../../outputs/analisis_sensibilidad_faltantes/resumen_efecto_imputacion.csv), [notebook 08](../../notebooks/08_umbrales_sensibilidad.ipynb). |
| 8 | 5 | [exploracion_completa/09_correlacion_grupo_A.png](exploracion_completa/09_correlacion_grupo_A.png) | Correlación de las variables del grupo A antes y después de la imputación descriptiva. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb), comparación de correlaciones. |
| 9 | 5 | [exploracion_completa/10_correlacion_grupo_B.png](exploracion_completa/10_correlacion_grupo_B.png) | Correlación de las variables del grupo B antes y después de la imputación descriptiva. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb). |
| 10 | 5 | [exploracion_completa/11_correlacion_grupo_C.png](exploracion_completa/11_correlacion_grupo_C.png) | Correlación de las variables del grupo C antes y después de la imputación descriptiva. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb), comparación de correlaciones. |
| 11 | 5 | [exploracion_completa/12_correlacion_grupo_D.png](exploracion_completa/12_correlacion_grupo_D.png) | Correlación de las variables del grupo D antes y después de la imputación descriptiva. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb), comparación de correlaciones. |
| 12 | 5 | [exploracion_completa/13_boxplots_antes_imputar.png](exploracion_completa/13_boxplots_antes_imputar.png) | Diagramas de caja de seis variables numéricas antes de imputar. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb), diagramas de caja. |
| 13 | 5 | [exploracion_completa/14_boxplots_despues_imputar.png](exploracion_completa/14_boxplots_despues_imputar.png) | Diagramas de caja de seis variables numéricas después de imputar. Fuente: elaboración propia; [notebook 04](../../notebooks/04_exploracion_completa.ipynb), diagramas de caja. |
| 14 | 7 | [16_baselines_desarrollo_completo.png](16_baselines_desarrollo_completo.png) | Comparación de ROC-AUC y Average Precision de los baselines sobre el conjunto de desarrollo completo. Fuente: elaboración propia; [notebook 06](../../notebooks/06_entrenamiento_busqueda_inicial.ipynb) y [resultados de los baselines](../../outputs/entrenamiento_26_variables/resumen_modelos_validacion.csv). |
| 15 | 9 | [optimizacion/busqueda_inicial_26.png](optimizacion/busqueda_inicial_26.png) | Primera búsqueda aleatoria con 26 características. Fuente: elaboración propia; [notebook 06](../../notebooks/06_entrenamiento_busqueda_inicial.ipynb) y [resultados de la búsqueda inicial](../../outputs/hiperparametros_randomized_26_variables/comparacion_por_anio.csv). |
| 16 | 9 | [optimizacion/01_comparacion_modelos.png](optimizacion/01_comparacion_modelos.png) | Comparación de configuraciones anteriores y mejores configuraciones de la optimización final. Fuente: elaboración propia; [notebook 09](../../notebooks/09_optimizacion_final.ipynb) y [comparación de configuraciones](../../outputs/optimizacion_final/comparacion_anterior_vs_mejor.csv). |
| 17 | 9 | [optimizacion/02_estabilidad_roc_por_anio.png](optimizacion/02_estabilidad_roc_por_anio.png) | Estabilidad temporal de las configuraciones finales. Fuente: elaboración propia; [notebook 09](../../notebooks/09_optimizacion_final.ipynb) y [resultados por configuración y corte](../../outputs/optimizacion_final/resultados_por_configuracion_y_corte.csv). |
| 18 | 10 | [calibracion/01_curvas_calibracion.png](calibracion/01_curvas_calibracion.png) | Comparación temporal de los métodos de calibración. Fuente: elaboración propia; [notebook 10](../../notebooks/10_calibracion.ipynb) y [métricas por modelo, método y año](../../outputs/calibracion_probabilidades/metricas_por_modelo_metodo_anio.csv). |
| 19 | 10 | [evaluacion_final/01_curvas_roc_test.png](evaluacion_final/01_curvas_roc_test.png) | Curvas ROC de los tres modelos finales. Fuente: elaboración propia; [notebook 11](../../notebooks/11_evaluacion_final.ipynb) y [métricas finales](../../outputs/evaluacion_final/metricas_finales_test.csv). |
| 20 | 10 | [evaluacion_final/02_curvas_precision_recall_test.png](evaluacion_final/02_curvas_precision_recall_test.png) | Curvas Precision-Recall de los tres modelos finales. Fuente: elaboración propia; [notebook 11](../../notebooks/11_evaluacion_final.ipynb) y [métricas finales](../../outputs/evaluacion_final/metricas_finales_test.csv). |
| 21 | 10 | [evaluacion_final/03_calibracion_test.png](evaluacion_final/03_calibracion_test.png) | Calibración de las probabilidades finales en test histórico. Fuente: elaboración propia; [notebook 11](../../notebooks/11_evaluacion_final.ipynb). |
| 22 | 10 | [evaluacion_final/04_calidad_probabilidades_test.png](evaluacion_final/04_calidad_probabilidades_test.png) | Comparación final de Brier Score y Log Loss. Fuente: elaboración propia; [notebook 11](../../notebooks/11_evaluacion_final.ipynb) y [métricas finales](../../outputs/evaluacion_final/metricas_finales_test.csv). |
| 23 | 12 | [analisis_errores/01_errores_por_anio.png](analisis_errores/01_errores_por_anio.png) | Falsos positivos y falsos negativos por año de prueba. Fuente: elaboración propia; [notebook 11](../../notebooks/11_evaluacion_final.ipynb) y [métricas por segmento](../../outputs/analisis_errores/metricas_por_segmento.csv). |
| 24 | 12 | [analisis_errores/02_recall_por_plazo.png](analisis_errores/02_recall_por_plazo.png) | Recall de HistGradientBoosting según el plazo del préstamo. Fuente: elaboración propia; [notebook 11](../../notebooks/11_evaluacion_final.ipynb) y [métricas por segmento](../../outputs/analisis_errores/metricas_por_segmento.csv). |

## Figuras del Anexo A

Estas dos figuras corresponden a comparaciones sobre el desarrollo completo documentadas en el notebook 07. A.1 compara tratamientos de faltantes en HistGradientBoosting; A.2 compara mediana e indicadores en regresión logística y SVM lineal. Su numeración es independiente de la secuencia principal.

| Figura | Archivo | Contenido y procedencia |
| --- | --- | --- |
| A.1 | [seleccion_variables/comparacion_imputacion.png](seleccion_variables/comparacion_imputacion.png) | Comparación de estrategias de imputación en HistGradientBoosting. Fuente: elaboración propia; [notebook 07](../../notebooks/07_comparacion_variables_imputacion.ipynb) y [resultados de comparación de imputación](../../outputs/comparacion_imputacion/resumen_promedios_y_variabilidad.csv). |
| A.2 | [seleccion_variables/comparacion_lineales.png](seleccion_variables/comparacion_lineales.png) | Comparación de estrategias de faltantes en regresión logística y SVM lineal. Fuente: elaboración propia; [notebook 07](../../notebooks/07_comparacion_variables_imputacion.ipynb) y [resultados de comparación de modelos lineales](../../outputs/comparacion_lineales/resumen_promedios_y_variabilidad.csv). |

## Orden oficial de notebooks

- [01_exploracion_inicial_baseline.ipynb](../../notebooks/01_exploracion_inicial_baseline.ipynb)
- [02_arbol_decision.ipynb](../../notebooks/02_arbol_decision.ipynb)
- [03_gradient_boosting.ipynb](../../notebooks/03_gradient_boosting.ipynb)
- [04_exploracion_completa.ipynb](../../notebooks/04_exploracion_completa.ipynb)
- [05_svm_muestra.ipynb](../../notebooks/05_svm_muestra.ipynb)
- [06_entrenamiento_busqueda_inicial.ipynb](../../notebooks/06_entrenamiento_busqueda_inicial.ipynb)
- [07_comparacion_variables_imputacion.ipynb](../../notebooks/07_comparacion_variables_imputacion.ipynb)
- [08_umbrales_sensibilidad.ipynb](../../notebooks/08_umbrales_sensibilidad.ipynb)
- [09_optimizacion_final.ipynb](../../notebooks/09_optimizacion_final.ipynb)
- [10_calibracion.ipynb](../../notebooks/10_calibracion.ipynb)
- [11_evaluacion_final.ipynb](../../notebooks/11_evaluacion_final.ipynb)

Los notebooks 01, 02, 03 y 05 documentan los experimentos iniciales. Sus figuras no se presentan como evidencia del conjunto completo en el informe definitivo. Las figuras de las secciones 5 y 7 proceden, respectivamente, de la exploración completa del notebook 04 y de los resultados de referencia del notebook 06.

## Recursos conservados sin insertar

Los siguientes PNG permanecen disponibles para consulta histórica o complementaria. No forman parte del inventario de evidencia visual del informe definitivo. Las figuras de la raíz sobre la muestra, las correlaciones de `exploracion/` y las de `gradient_boosting_220000/` pertenecen a los experimentos iniciales; no deben confundirse con las imágenes de `exploracion_completa/` ni con las evaluaciones finales.

| Archivo no insertado |
| --- |
| [01_distribucion_objetivo.png](01_distribucion_objetivo.png) |
| [02_default_por_anio.png](02_default_por_anio.png) |
| [03_horizonte_observacion.png](03_horizonte_observacion.png) |
| [04_faltantes.png](04_faltantes.png) |
| [05_variables_clave.png](05_variables_clave.png) |
| [06_categoricas.png](06_categoricas.png) |
| [07_outliers.png](07_outliers.png) |
| [08_curvas_baseline.png](08_curvas_baseline.png) |
| [09_coeficientes_baseline.png](09_coeficientes_baseline.png) |
| [10_arbol_complejidad.png](10_arbol_complejidad.png) |
| [11_arbol_reglas.png](11_arbol_reglas.png) |
| [12_arbol_importancia.png](12_arbol_importancia.png) |
| [13_comparacion_roc.png](13_comparacion_roc.png) |
| [14_boosting_busqueda.png](14_boosting_busqueda.png) |
| [15_boosting_importancia.png](15_boosting_importancia.png) |
| [16_boosting_comparacion_roc.png](16_boosting_comparacion_roc.png) |
| [calibracion/02_brier_score.png](calibracion/02_brier_score.png) |
| [calibracion/03_log_loss.png](calibracion/03_log_loss.png) |
| [calibracion/04_roc_auc.png](calibracion/04_roc_auc.png) |
| [calibracion/05_pr_auc.png](calibracion/05_pr_auc.png) |
| [calibracion/06_distribucion_probabilidades.png](calibracion/06_distribucion_probabilidades.png) |
| [exploracion/correlacion_antes_A.png](exploracion/correlacion_antes_A.png) |
| [exploracion/correlacion_antes_B.png](exploracion/correlacion_antes_B.png) |
| [exploracion/correlacion_antes_C.png](exploracion/correlacion_antes_C.png) |
| [exploracion/correlacion_antes_D.png](exploracion/correlacion_antes_D.png) |
| [exploracion/correlacion_despues_A.png](exploracion/correlacion_despues_A.png) |
| [exploracion/correlacion_despues_B.png](exploracion/correlacion_despues_B.png) |
| [exploracion/correlacion_despues_C.png](exploracion/correlacion_despues_C.png) |
| [exploracion/correlacion_despues_D.png](exploracion/correlacion_despues_D.png) |
| [gradient_boosting_220000/14_boosting_busqueda.png](gradient_boosting_220000/14_boosting_busqueda.png) |
| [gradient_boosting_220000/15_boosting_importancia.png](gradient_boosting_220000/15_boosting_importancia.png) |
| [gradient_boosting_220000/16_boosting_comparacion_roc.png](gradient_boosting_220000/16_boosting_comparacion_roc.png) |
| [optimizacion/03_roc_vs_pr_configuraciones.png](optimizacion/03_roc_vs_pr_configuraciones.png) |
| [seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_rev_tl_op.png](seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_rev_tl_op.png) |
| [seleccion_variables/sensibilidad/02_distribucion_mo_sin_rcnt_rev_tl_op.png](seleccion_variables/sensibilidad/02_distribucion_mo_sin_rcnt_rev_tl_op.png) |
| [seleccion_variables/sensibilidad/02_distribucion_mths_since_last_major_derog.png](seleccion_variables/sensibilidad/02_distribucion_mths_since_last_major_derog.png) |
| [seleccion_variables/sensibilidad/04_metricas_temporales_gradient_boosting.png](seleccion_variables/sensibilidad/04_metricas_temporales_gradient_boosting.png) |
| [seleccion_variables/umbrales/02_pr_auc_promedio.png](seleccion_variables/umbrales/02_pr_auc_promedio.png) |
| [seleccion_variables/umbrales/04_roc_auc_por_anio.png](seleccion_variables/umbrales/04_roc_auc_por_anio.png) |
| [seleccion_variables/umbrales/05_pr_auc_por_anio.png](seleccion_variables/umbrales/05_pr_auc_por_anio.png) |

Balance del inventario: 26 imágenes insertadas (24 principales y 2 del anexo); 40 PNG conservados sin insertar. Las rutas y fuentes se contrastaron con el informe y los archivos existentes. Esta revisión documental no ejecuta ni modifica experimentos.
