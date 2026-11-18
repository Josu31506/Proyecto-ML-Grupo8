# Guion de presentación final

## 1. Problema y uso previsto

- Estimar, por préstamo aceptado, el riesgo de terminar como `Charged Off`.
- El score ordena riesgo y sirve como apoyo; no es una regla automática de aprobación.

## 2. Datos y población analizada

- Lending Club 2007–2018; estados finales `Fully Paid` y `Charged Off`.
- 1,345,310 préstamos elegibles en el conjunto completo.
- Test 2017–2018: 938,821 préstamos totales; 225,611 con uno de los dos estados finales.
- Visual sugerido: distribución de estados por año y proporción de préstamos aún no resueltos.

## 3. Preparación y prevención de fuga de información

- Excluir identificadores y variables posteriores al otorgamiento.
- Separación temporal; imputadores y codificación ajustados solo sobre la base de cada corte.
- En la búsqueda final, seleccionar con ≤50 % de faltantes por tramo base; la lista cambia entre años.
- Diferenciar claramente la regla histórica de 10 % de la exploración y la regla final de 50 %.

## 4. Evolución de los modelos

- Referencia mayoritaria y baseline de regresión logística.
- Árbol de decisión y Gradient Boosting histórico (muestra pequeña, solo como exploración).
- SVM lineal y HistGradientBoosting optimizado en el flujo final.
- La rama inicial de Gradient Boosting no se compara directamente: usó 5,002 filas y un corte, frente a cientos de miles y cuatro cortes en el flujo final.

## 5. Validación y búsqueda

- Cortes expansivos: base hasta 2012→2013; 2013→2014; 2014→2015; 2015→2016.
- Randomized Search en ROC-AUC; AP y estabilidad anual como apoyo.
- Configuración final: la mejor entre las probadas; no es un óptimo universal.
- Mencionar que test fue visto en experimentos históricos anteriores.

## 6. Evaluación final

| Modelo calibrado | ROC-AUC | AP | Brier | Log Loss |
|---|---:|---:|---:|---:|
| Regresión logística | 0.6926 | 0.3604 | 0.1559 | 0.4850 |
| SVM lineal | 0.6918 | 0.3599 | 0.1560 | 0.4868 |
| HistGradientBoosting | **0.7032** | **0.3779** | **0.1536** | **0.4743** |

Visual sugerido: ROC y Precision-Recall, acompañadas por prevalencia positiva de 21.28 %.

## 7. Errores por segmentos

- Umbral descriptivo fijo de 0.50, no ajustado en test.
- HistGradientBoosting: recall 0.131, precision 0.493; 41,746 falsos negativos y 6,459 falsos positivos.
- Diferencia por plazo: recall 0.010 en 36 meses y 0.354 en 60 meses; revisar calibración y base por segmento antes de interpretar.

## 8. Limitaciones y siguientes pasos

- Los estados finales no traen fecha de maduración; los préstamos activos se excluyen.
- El test condicionado a desenlaces resueltos no representa todos los préstamos emitidos.
- Sin análisis completo de equidad ni decisión de costo de errores.
- Siguiente: validar maduración con fecha de corte, revisar diferencias por grupos y calcular interpretabilidad del modelo final en validación.
