# Inventario de figuras del informe final

La numeración corresponde a `reports/informe_final.md`. Las figuras 33–42 pertenecen al bloque independiente «Anexo A. Evidencia visual complementaria»; el inventario se presenta en el Anexo B, después de los quince apartados numerados. Los archivos originales en outputs y las salidas incrustadas de notebooks permanecen intactos. No se entrenaron modelos ni se generaron nuevos resultados experimentales.

## Imágenes utilizadas

| Figura | Sección | Archivo | Título y procedencia |
| --- | --- | --- | --- |
| 1 | 3 | [exploracion/distribucion_objetivo_completo.png](exploracion/distribucion_objetivo_completo.png) | Distribución de los dos desenlaces en toda la población elegible; [notebook 04](../../notebooks/04_exploracion_completa.ipynb), salida guardada de distribución del objetivo |
| 2 | 4 | [04_faltantes.png](04_faltantes.png) | Faltantes en la muestra antes de la exclusión conceptual; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb) y [auditoría inicial](../../outputs/auditoria.json) |
| 3 | 4 | [seleccion_variables/sensibilidad/01_disponibilidad_anual.png](seleccion_variables/sensibilidad/01_disponibilidad_anual.png) | Disponibilidad de variables a través de los años; [notebook 07](../../notebooks/07_umbrales_sensibilidad.ipynb) y [disponibilidad anual](../../outputs/analisis_sensibilidad_faltantes/disponibilidad_anual_variables.csv) |
| 4 | 4 | [seleccion_variables/comparacion_conjuntos.png](seleccion_variables/comparacion_conjuntos.png) | Comparación de conjuntos y de indicadores de faltantes; [notebook 06](../../notebooks/06_comparacion_variables_imputacion.ipynb), comparación temporal de conjuntos |
| 5 | 4 | [seleccion_variables/umbrales/01_roc_auc_promedio.png](seleccion_variables/umbrales/01_roc_auc_promedio.png) | ROC-AUC medio según el límite de faltantes; [notebook 07](../../notebooks/07_umbrales_sensibilidad.ipynb) y comparación de umbrales |
| 6 | 4 | [seleccion_variables/umbrales/03_variables_por_corte.png](seleccion_variables/umbrales/03_variables_por_corte.png) | Número de variables admitidas por corte y umbral; [conteo de variables](../../outputs/comparacion_umbrales_faltantes/conteo_variables_por_corte_y_umbral.csv), notebook 07 |
| 7 | 4 | [seleccion_variables/sensibilidad/03_delta_roc_anual.png](seleccion_variables/sensibilidad/03_delta_roc_anual.png) | Cambio anual de ROC-AUC al pasar de 50% a 75%; [diferencias anuales](../../outputs/analisis_sensibilidad_faltantes/metricas_anuales_50_vs_75.csv), notebook 07 |
| 8 | 4 | [seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_il_acct.png](seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_il_acct.png) | Distribución observada e imputada de la antigüedad de crédito a plazos; [resumen del efecto de imputación](../../outputs/analisis_sensibilidad_faltantes/resumen_efecto_imputacion.csv), notebook 07 |
| 9 | 5 | [02_default_por_anio.png](02_default_por_anio.png) | Distribución temporal y proporción de incumplimiento en la muestra inicial; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb) y auditoría inicial |
| 10 | 5 | [03_horizonte_observacion.png](03_horizonte_observacion.png) | Plazo y seguimiento disponible en la exploración inicial; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), muestra inicial |
| 11 | 5 | [05_variables_clave.png](05_variables_clave.png) | Distribuciones de variables numéricas en el entrenamiento inicial; [src/run_baseline.py](../../src/run_baseline.py), figuras de la muestra inicial |
| 12 | 5 | [06_categoricas.png](06_categoricas.png) | Resultados según categorías en la muestra inicial; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb) y generador de figuras del baseline |
| 13 | 5 | [exploracion/correlacion_antes_C.png](exploracion/correlacion_antes_C.png) | Grupo C: asociación con el objetivo antes de completar faltantes; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), salida incrustada de la sección de correlación previa |
| 14 | 5 | [exploracion/correlacion_despues_C.png](exploracion/correlacion_despues_C.png) | Grupo C: asociación con el objetivo después de completar faltantes; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), salida incrustada de correlación posterior |
| 15 | 5 | [07_outliers.png](07_outliers.png) | Diagramas de caja de variables numéricas del entrenamiento inicial; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb) y figura existente de outliers |
| 16 | 7 | [08_curvas_baseline.png](08_curvas_baseline.png) | Curvas y métricas de referencia del baseline inicial; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb) y [métricas del baseline](../../outputs/metrics.json) |
| 17 | 8 | [10_arbol_complejidad.png](10_arbol_complejidad.png) | Complejidad y generalización del árbol inicial; [notebook 02](../../notebooks/02_arbol_decision.ipynb) y [búsqueda del árbol](../../outputs/busqueda_arbol.csv) |
| 18 | 8 | [11_arbol_reglas.png](11_arbol_reglas.png) | Reglas superiores del árbol regularizado; [reglas guardadas](../../outputs/reglas_arbol.txt), notebook 02 |
| 19 | 8 | [gradient_boosting_220000/16_boosting_comparacion_roc.png](gradient_boosting_220000/16_boosting_comparacion_roc.png) | Comparación homogénea con Gradient Boosting en 220,003 préstamos; [métricas de la ejecución homogénea](../../outputs/gradient_boosting_220000/metrics_boosting.json), notebook 03 |
| 20 | 9 | [gradient_boosting_220000/14_boosting_busqueda.png](gradient_boosting_220000/14_boosting_busqueda.png) | Búsqueda de Gradient Boosting sobre la muestra homogénea; [busqueda_boosting.csv](../../outputs/gradient_boosting_220000/busqueda_boosting.csv), notebook 03 |
| 21 | 9 | [optimizacion/busqueda_inicial_26.png](optimizacion/busqueda_inicial_26.png) | Primera búsqueda aleatoria con 26 variables; [notebook 05](../../notebooks/05_entrenamiento_busqueda_inicial.ipynb), salida guardada de comparación |
| 22 | 9 | [optimizacion/01_comparacion_modelos.png](optimizacion/01_comparacion_modelos.png) | Modelos originales y mejores configuraciones del espacio final; [comparacion_anterior_vs_mejor.csv](../../outputs/optimizacion_final/comparacion_anterior_vs_mejor.csv), notebook 08 |
| 23 | 9 | [optimizacion/02_estabilidad_roc_por_anio.png](optimizacion/02_estabilidad_roc_por_anio.png) | Estabilidad temporal de las configuraciones finales; [resultados por configuración y corte](../../outputs/optimizacion_final/resultados_por_configuracion_y_corte.csv), notebook 08 |
| 24 | 10 | [calibracion/01_curvas_calibracion.png](calibracion/01_curvas_calibracion.png) | Comparación temporal de métodos de calibración; [notebook 09](../../notebooks/09_calibracion.ipynb) y [métricas por método y año](../../outputs/calibracion_probabilidades/metricas_por_modelo_metodo_anio.csv) |
| 25 | 10 | [evaluacion_final/01_curvas_roc_test.png](evaluacion_final/01_curvas_roc_test.png) | Curvas ROC de los tres modelos finales en test histórico; [notebook 10](../../notebooks/10_evaluacion_final.ipynb) y métricas finales guardadas |
| 26 | 10 | [evaluacion_final/02_curvas_precision_recall_test.png](evaluacion_final/02_curvas_precision_recall_test.png) | Curvas Precision–Recall de los modelos finales; [evaluación histórica final](../../outputs/evaluacion_final/metricas_finales_test.csv), notebook 10 |
| 27 | 10 | [evaluacion_final/03_calibracion_test.png](evaluacion_final/03_calibracion_test.png) | Calibración de las probabilidades finales en test histórico; [notebook 10](../../notebooks/10_evaluacion_final.ipynb), curva guardada de calibración |
| 28 | 10 | [evaluacion_final/04_calidad_probabilidades_test.png](evaluacion_final/04_calidad_probabilidades_test.png) | Comparación final de Brier y Log Loss; [métricas finales de test](../../outputs/evaluacion_final/metricas_finales_test.csv), notebook 10 |
| 29 | 11 | [09_coeficientes_baseline.png](09_coeficientes_baseline.png) | Coeficientes de la regresión logística inicial; [coeficientes_baseline.csv](../../outputs/coeficientes_baseline.csv), muestra del notebook 01 |
| 30 | 11 | [gradient_boosting_220000/15_boosting_importancia.png](gradient_boosting_220000/15_boosting_importancia.png) | Importancia por permutación del Gradient Boosting de la muestra; [importancia_boosting.csv](../../outputs/gradient_boosting_220000/importancia_boosting.csv), experimento de 220,003 préstamos |
| 31 | 12 | [analisis_errores/01_errores_por_anio.png](analisis_errores/01_errores_por_anio.png) | Errores y métricas descriptivas por año de test; [notebook 10](../../notebooks/10_evaluacion_final.ipynb) y métricas por segmento |
| 32 | 12 | [analisis_errores/02_recall_por_plazo.png](analisis_errores/02_recall_por_plazo.png) | Detección de incumplimientos según plazo; [metricas_por_segmento.csv](../../outputs/analisis_errores/metricas_por_segmento.csv), notebook 10 |
| 33 | Anexo A | [exploracion/correlacion_antes_A.png](exploracion/correlacion_antes_A.png) | Grupo A: correlación con el objetivo antes de imputar; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), correlación numérica en entrenamiento de la muestra |
| 34 | Anexo A | [exploracion/correlacion_despues_A.png](exploracion/correlacion_despues_A.png) | Grupo A: correlación con el objetivo después de imputar; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), correlación numérica en entrenamiento de la muestra |
| 35 | Anexo A | [exploracion/correlacion_antes_B.png](exploracion/correlacion_antes_B.png) | Grupo B: correlación con el objetivo antes de imputar; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), correlación numérica en entrenamiento de la muestra |
| 36 | Anexo A | [exploracion/correlacion_despues_B.png](exploracion/correlacion_despues_B.png) | Grupo B: correlación con el objetivo después de imputar; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), correlación numérica en entrenamiento de la muestra |
| 37 | Anexo A | [exploracion/correlacion_antes_D.png](exploracion/correlacion_antes_D.png) | Grupo D: correlación con el objetivo antes de imputar; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), correlación numérica en entrenamiento de la muestra |
| 38 | Anexo A | [exploracion/correlacion_despues_D.png](exploracion/correlacion_despues_D.png) | Grupo D: correlación con el objetivo después de imputar; [notebook 01](../../notebooks/01_exploracion_inicial_baseline.ipynb), correlación numérica en entrenamiento de la muestra |
| 39 | Anexo A | [seleccion_variables/comparacion_imputacion.png](seleccion_variables/comparacion_imputacion.png) | Mediana, indicadores y tratamiento nativo de ausentes; [notebook 06](../../notebooks/06_comparacion_variables_imputacion.ipynb) y [comparación de imputación](../../outputs/comparacion_imputacion/resumen_promedios_y_variabilidad.csv) |
| 40 | Anexo A | [seleccion_variables/comparacion_lineales.png](seleccion_variables/comparacion_lineales.png) | Estrategias de faltantes en los modelos lineales; [notebook 06](../../notebooks/06_comparacion_variables_imputacion.ipynb) y [comparación lineal](../../outputs/comparacion_lineales/resumen_promedios_y_variabilidad.csv) |
| 41 | Anexo A | [optimizacion/03_roc_vs_pr_configuraciones.png](optimizacion/03_roc_vs_pr_configuraciones.png) | Relación entre ROC-AUC y AP de las configuraciones evaluadas; [notebook 08](../../notebooks/08_optimizacion_final.ipynb), espacio aleatorio final |
| 42 | Anexo A | [calibracion/06_distribucion_probabilidades.png](calibracion/06_distribucion_probabilidades.png) | Distribución de probabilidades en la comparación de calibración; [notebook 09](../../notebooks/09_calibracion.ipynb), predicciones de validación |

## Imágenes conservadas sin insertar en el informe

Todas permanecen disponibles; no se eliminaron archivos históricos.

| Archivo | Motivo de no incorporación |
| --- | --- |
| [01_distribucion_objetivo.png](01_distribucion_objetivo.png) | Distribución de la muestra; el cuerpo prioriza la figura global del notebook 04 y documenta los conteos de la muestra en la sección 7. |
| [12_arbol_importancia.png](12_arbol_importancia.png) | Importancia exploratoria del árbol; se priorizan reglas y complejidad, y la interpretación del ensamble homogéneo. No representa al modelo final. |
| [13_comparacion_roc.png](13_comparacion_roc.png) | Comparación inicial redundante con las curvas del baseline y la comparación homogénea de 220,003 préstamos. |
| [14_boosting_busqueda.png](14_boosting_busqueda.png) | Ejecución preliminar distinta de 5,002 observaciones; no mezclar con la comparación homogénea. Procedencia advertida en sección 8.3. |
| [15_boosting_importancia.png](15_boosting_importancia.png) | Importancia del boosting preliminar, de otra muestra; se utiliza la de gradient_boosting_220000 con procedencia homogénea. |
| [16_boosting_comparacion_roc.png](16_boosting_comparacion_roc.png) | Comparación preliminar no homogénea con los otros outputs históricos; no utilizada como evidencia del ganador. |
| [calibracion/02_brier_score.png](calibracion/02_brier_score.png) | Brier por método está tabulado en sección 10.1; se conservan curvas y pérdidas finales como figuras principales. |
| [calibracion/03_log_loss.png](calibracion/03_log_loss.png) | Log Loss por método está tabulada en sección 10.1; evita repetir el mismo contraste. |
| [calibracion/04_roc_auc.png](calibracion/04_roc_auc.png) | ROC-AUC de calibración tabulado y explicado, incluida la separación respecto de la búsqueda. |
| [calibracion/05_pr_auc.png](calibracion/05_pr_auc.png) | AP de calibración tabulada; figura adicional no añade una comparación diferente. |
| [seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_rev_tl_op.png](seleccion_variables/sensibilidad/02_distribucion_mo_sin_old_rev_tl_op.png) | Diagnóstico adicional del mismo efecto de imputación; se muestra un ejemplo completo y se enlaza la tabla original. |
| [seleccion_variables/sensibilidad/02_distribucion_mo_sin_rcnt_rev_tl_op.png](seleccion_variables/sensibilidad/02_distribucion_mo_sin_rcnt_rev_tl_op.png) | Diagnóstico adicional de antigüedad; evita repetir la explicación del efecto de imputación. |
| [seleccion_variables/sensibilidad/02_distribucion_mths_since_last_major_derog.png](seleccion_variables/sensibilidad/02_distribucion_mths_since_last_major_derog.png) | Diagnóstico adicional de una variable de alta ausencia; se conserva para consulta, sin atribuir semántica a sus vacíos. |
| [seleccion_variables/sensibilidad/04_metricas_temporales_gradient_boosting.png](seleccion_variables/sensibilidad/04_metricas_temporales_gradient_boosting.png) | La tabla de sensibilidad y el gráfico delta anual ya muestran el contraste por años. |
| [seleccion_variables/umbrales/02_pr_auc_promedio.png](seleccion_variables/umbrales/02_pr_auc_promedio.png) | Valores AP incorporados en la tabla de umbrales; gráfico adicional redundante con ROC medio y conteos. |
| [seleccion_variables/umbrales/04_roc_auc_por_anio.png](seleccion_variables/umbrales/04_roc_auc_por_anio.png) | La sensibilidad anual y los resultados tabulados desarrollan el cambio relevante; se conserva para consulta. |
| [seleccion_variables/umbrales/05_pr_auc_por_anio.png](seleccion_variables/umbrales/05_pr_auc_por_anio.png) | Diagnóstico complementario de AP temporal; tabla y discusión distinguen discriminación y prevalencia. |

**Balance:** 59 imágenes disponibles; 42 incorporadas (32 en el cuerpo y 10 en el anexo); 17 conservadas sin insertar.

## Organización y fuentes

Las imágenes históricas de la raíz y de gradient_boosting_220000 y analisis_errores se conservaron en sus rutas. Las de outputs se copiaron a seleccion_variables, optimizacion, calibracion y evaluacion_final para mantener compatibilidad. Se extrajeron 13 PNG de salidas guardadas de notebooks: ocho mapas de correlación de 01, la distribución global de 04, una comparación de búsqueda de 05 y tres comparaciones de 06. No se copiaron imágenes idénticas dentro de esta carpeta.

## Comprobaciones de la revisión documental

| Control | Resultado |
| --- | --- |
| Apartados principales del informe | Exactamente 15, en el orden solicitado |
| Tablas del informe | 19, renderizadas sin desbordamiento en la vista previa |
| Imágenes insertadas y pies | 42, con numeración consecutiva, título, fuente e interpretación |
| Disponibilidad de imágenes en vista previa | 42 cargadas; ninguna rota |
| Enlaces locales del informe y de este inventario | Todos resuelven a archivos o carpetas existentes |
| Duplicados de imágenes en reports/figures | Ninguno con contenido binario idéntico |
| Cifras de métricas contrastadas automáticamente | 68 valores de calibración, optimización y test; 55 valores y 11 tamaños por segmento |
| Conservación de archivos protegidos | 187 archivos de notebooks, src, tests, outputs, models, data, README principal, requirements y el ZIP local con SHA-256 sin cambios en la corrección final |
| Ejecuciones de modelos durante la revisión | Ninguna; solo lectura de evidencia y renderizado documental |
| Referencias a notebooks retirados | No hay enlaces a notebooks 11 o 12 |
| Vista previa | HTML temporal fuera del repositorio; revisados inicio, métricas y matrices de confusión |

## Correcciones principales y asuntos pendientes

El informe reemplaza las referencias antiguas al notebook 12 por el notebook 10, reconoce los tres modelos serializados, integra los resultados ya ejecutados de boosting sobre 220,003 préstamos y separa esa muestra del experimento preliminar de 5,002. Actualiza los comandos al pipeline existente y distingue los resultados históricos de una nueva reproducción todavía no confirmada.

También corrige la interpretación del límite de faltantes: 50% es una elección conservadora, mientras 75% obtuvo mayor ROC-AUC promedio en la comparación guardada. Señala que la aparente agrupación bootstrap por prestatario no está verificada cuando member_id está vacío y que los ajustes de calibración utilizan otros años base que la búsqueda. No se modificaron cifras históricas para ocultar esas diferencias.

Pendientes científicos: paridad real de predicciones del pipeline reciente, fecha efectiva de disponibilidad de etiquetas, evaluación prospectiva independiente, interpretación del modelo final y auditoría de equidad. Se explican en las secciones 13–14. La elaboración inicial no publicó cambios. Esta corrección final autoriza un commit y push limitados al informe y sus recursos documentales en docs/entrega-final; no incluye PR ni merge a main. El archivo local revision_informe.zip queda fuera de la publicación.

## Corrección final para publicación

La sección 15.1 exige cambiar a docs/entrega-final después de clonar y presenta las pruebas antes de entrenar; los comandos se contrastaron con el parser y el localizador del código vigente. La sección 3.1 identifica a wordsforthewise como publicador en Kaggle y deja explícito que no fue posible verificar la licencia o condiciones de reutilización en la fuente primaria durante la consulta del 17 de noviembre de 2026 (página sin contenido legible y consultas HTTP 404). No se atribuye una licencia basada en copias de terceros. Esta URL externa se conserva como referencia de procedencia, sin confundirla con un enlace de descarga cuya disponibilidad esté confirmada.

Los anexos y referencias ya no son subsecciones 15.7–15.9: se distinguen mediante títulos en negrita y separadores, sin añadir apartados numerados. Las seis interpretaciones de las figuras 33–38 describen variables y diferencias concretas de sus imágenes, conservando los PNG y sus valores. Se mantienen las 19 tablas, 42 figuras y 17 imágenes sin insertar.
