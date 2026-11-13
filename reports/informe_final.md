# Informe final — Predicción de incumplimiento en Lending Club

**Grupo 8 · UTEC · Machine Learning**  
**Estado del informe:** borrador basado en artefactos locales ejecutados. No sustituye una revisión de maduración de etiquetas ni la aprobación del profesor.

## 1. Resumen ejecutivo

Se compararon estrategias de referencia y modelos para ordenar préstamos según riesgo de `Charged Off`. En la evaluación final guardada, HistGradientBoosting obtuvo los mejores resultados entre los tres modelos calibrados: ROC-AUC 0.7032, Average Precision 0.3779, Brier 0.1536 y Log Loss 0.4743 sobre 225,611 préstamos resueltos de 2017–2018. Las diferencias son moderadas; no prueban que el modelo sea útil para aprobar o rechazar préstamos. El test contiene solo casos con desenlace final observado y los registros no indican cuándo se conoció ese desenlace.

## 2. Problema

La pregunta del proyecto es: dado un préstamo aceptado y usando información asociada al otorgamiento, ¿qué probabilidad estima el modelo para que termine como `Charged Off` en vez de `Fully Paid`? La predicción es por préstamo individual. Se plantea como apoyo para ordenar riesgo, no como decisión automática sobre una persona.

## 3. Datos

Se utiliza Lending Club Loan Data 2007–2018, archivo `accepted_2007_to_2018Q4.csv`, con 2,260,701 filas y 151 columnas en la auditoría de la primera etapa. El flujo de evaluación final lee candidatas y variables de control desde el archivo original. La fuente es la distribución del conjunto de Lending Club referenciada en la propuesta y en `data/README.md`. El CSV no debe incluirse en Git.

La variable binaria se deriva de estados finales exactos: `Fully Paid = 0`, `Charged Off = 1`. En el archivo completo hay 1,345,310 préstamos de esos dos estados. `Current`, atrasos y periodo de gracia se excluyen porque no son desenlaces finales; `Default` (40 filas) se dejó fuera como etiqueta distinta. El conjunto final no representa préstamos rechazados.

## 4. Preparación

Se excluyen identificadores, texto fuera del alcance inicial, códigos constantes y variables conocidas después del otorgamiento (pagos recibidos, saldos posteriores, recuperaciones, FICO actualizado, dificultades y acuerdos). Las fechas del historial se convierten a antigüedad relativa cuando forman parte de candidatas. El escenario final excluye también el bloque de calificación/precio de Lending Club; su disponibilidad en el instante exacto de decisión requiere validación con el diccionario y el proceso operativo.

En la búsqueda final se calcula en cada tramo base la proporción de faltantes y se conservan variables con hasta 50 %. La selección fue 26 variables para validación 2013 y 60 para 2014, 2015 y 2016; el ajuste final seleccionó 60. Numéricas: mediana aprendida solo en la base; categóricas: `Sin información` y One-Hot. No se incluyeron indicadores de faltantes en la evaluación final. Las variables con extremos IQR no se eliminaron automáticamente; la regla sirvió para detectar casos que requieren revisión, no para filtrar observaciones sin justificación.

La cobertura cambia con el año. La regla 10 % de la etapa de exploración y el límite histórico 95 % no generaron los resultados definitivos: el pipeline final usa 50 % por tramo base (`src/etapas_finales.py`).

## 5. Exploración

La exploración inicial documenta distribución del objetivo, faltantes, relaciones, outliers y categorías con una muestra. La exploración final revisa el archivo completo y el patrón de faltantes antes de las particiones. Los activos y los préstamos sin desenlace final se excluyen del modelado, pero se conservan en auditorías de población. Las visualizaciones y tablas disponibles están en los notebooks 01 y 04.

## 6. Validación

La optimización usa cortes expansivos por año de otorgamiento: entrenar hasta 2012 y validar 2013; hasta 2013 y validar 2014; hasta 2014 y validar 2015; hasta 2015 y validar 2016. La selección de variables y las transformaciones se ajustan solo con la base de cada corte. Los datos 2017 en adelante se reservaron para la evaluación final actual, pero los experimentos históricos de baseline, árbol y reducción de dimensionalidad ya habían consultado métricas de prueba. Por ello, no afirmamos que el test nunca se haya inspeccionado.

Los estados guardados no incluyen la fecha individual en que el préstamo alcanzó el desenlace final. La separación por año de originación no demuestra que la etiqueta de cada préstamo del entrenamiento hubiera sido conocida en el momento histórico simulado. Se necesita fecha de corte/actualización y una auditoría de maduración antes de interpretar estas métricas como una prueba prospectiva estricta.

## 7. Referencias

La referencia mayoritaria obtiene ROC-AUC 0.5 y no detecta positivos por su regla de decisión. La regresión logística fue el baseline estadístico inicial. La primera etapa trabajó con una muestra y una preparación distinta, por lo que su ROC-AUC de validación 0.7121 no se compara como si fuera el mismo experimento de la evaluación final.

## 8. Modelos

El historial del repositorio conserva: clase mayoritaria, regresión logística, árbol de decisión, Gradient Boosting inicial, SVM lineal e HistGradientBoosting optimizado. El árbol individual y los ensambles de boosting comparten árboles como componentes, pero son estrategias de aprendizaje distintas; logística y SVM son dos clasificadores lineales con objetivos diferentes. Los tres modelos finales evaluados fueron regresión logística, SVM lineal e HistGradientBoosting.

## 9. Búsqueda de hiperparámetros

`09g_optimizacion_final.ipynb` probó configuraciones aleatorias en los cuatro cortes, manteniendo el test fuera de la búsqueda final. La mejor configuración disponible dentro de los espacios explorados obtuvo estos promedios no calibrados:

| Modelo | ROC-AUC promedio | Desv. ROC-AUC | AP promedio | Desv. AP | Variables originales promedio |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.7024 | 0.0237 | 0.3595 | 0.0709 | 51.5 |
| SVM lineal | 0.7022 | 0.0241 | 0.3586 | 0.0711 | 51.5 |
| HistGradientBoosting | **0.7105** | 0.0246 | **0.3695** | 0.0744 | 51.5 |

Las diferencias frente a las configuraciones de referencia fueron pequeñas: ROC-AUC +0.0001 para logística, 0 para SVM y +0.0001 para HistGradientBoosting. La búsqueda encontró las mejores configuraciones entre las evaluadas, no un óptimo universal. Los parámetros reproducibles están en `outputs/optimizacion_final/configuracion_reproducible.json`.

## 10. Evaluación final

La calibración se seleccionó por validación temporal. El ajuste final usa base hasta 2015, calibración 2016 y evaluación 2017–2018. Los tres modelos reportados usan calibración sigmoide.

| Modelo | ROC-AUC | AP (PR-AUC) | Brier | Log Loss |
|---|---:|---:|---:|---:|
| Regresión logística | 0.6926 | 0.3604 | 0.1559 | 0.4850 |
| SVM lineal | 0.6918 | 0.3599 | 0.1560 | 0.4868 |
| HistGradientBoosting | **0.7032** | **0.3779** | **0.1536** | **0.4743** |

En 2017–2018 hay 938,821 préstamos de todas las situaciones, de los cuales 225,611 tienen desenlace binario elegible. La prevalencia de `Charged Off` entre esos casos resueltos es 21.28 %. Para HistGradientBoosting, el intervalo bootstrap 95 % guardado fue 0.7009–0.7059 para ROC-AUC y 0.3746–0.3822 para AP; refleja variación de remuestreo dentro de esa muestra, no corrige la selección por maduración.

## 11. Interpretación

HistGradientBoosting supera modestamente a logística y SVM en ROC-AUC, AP, Brier y Log Loss del conjunto reportado. ROC-AUC describe ordenamiento; AP se enfoca más en la clase positiva; Brier y Log Loss evalúan la calidad de las probabilidades. Ninguna métrica, por sí sola, indica cuál sería la probabilidad “correcta” para cada préstamo ni traduce el score a una decisión rentable.

Los coeficientes de logística y las importancias del árbol disponibles pertenecen a experimentos anteriores. No se deben atribuir al HistGradientBoosting final. El artefacto no guarda el estimador final serializado ni una importancia por permutación calculada en validación con el pipeline final.

## 12. Análisis de errores

El notebook 12 usa las probabilidades finales guardadas y un umbral fijo de 0.50 únicamente para describir errores; no lo ajusta con test. Con HistGradientBoosting se obtuvieron 6,269 verdaderos positivos, 41,746 falsos negativos, 6,459 falsos positivos y 171,137 verdaderos negativos. A este umbral, precision fue 0.493 y recall 0.131. Por tanto, el buen ordenamiento global no significa que la regla 0.50 detecte la mayoría de incumplimientos.

El recall del grupo de préstamos a 36 meses fue 0.010 con umbral 0.50, frente a 0.354 para 60 meses; ROC-AUC por grupo fue 0.703 y 0.675, respectivamente. La tasa de incumplimiento fue 18.37 % y 30.16 %. Estas cifras describen diferencias importantes por segmento, pero no explican su causa ni equivalen a una política aprobada. El detalle por año, plazo, tramo de ingreso y tipo de solicitud se genera en `outputs/analisis_errores/`.

## 13. Sesgos, ética y limitaciones

- Solo se modelan préstamos aceptados; el conjunto no indica cómo habrían pagado los rechazados.
- Se condiciona a que el préstamo tenga un estado final. En los años recientes, se excluyen muchos préstamos activos, y los observados pueden representar principalmente desenlaces más rápidos.
- No se dispone del momento exacto en que se conoció cada resultado, por lo que el seguimiento histórico no se puede validar completamente.
- Los ingresos, la ubicación y los datos del historial pueden contener errores o reflejar diferencias históricas. Este proyecto no mide equidad por grupos sensibles.
- El umbral 0.50 no fue definido con costos de errores y no debe usarse para aprobar o rechazar automáticamente.
- Los resultados no prueban causalidad. Una característica predictiva no significa que cambiar esa característica cambie el riesgo.
- El test fue visto en análisis anteriores; aunque la calibración final y búsqueda reportadas se basaron en validación, la evaluación no es una prueba completamente virgen para todo el ciclo del proyecto.

## 14. Trabajo futuro

Confirmar la fecha de corte del conjunto y evaluar maduración por plazo; si se dispone de fechas, definir desenlaces conocidos a una fecha fija. Revisar cobertura y sesgo por grupos relevantes. Evaluar umbrales únicamente con validación y una función de costo acordada. Medir interpretabilidad del HistGradientBoosting con permutación en validación. Guardar el pipeline final reproducible y repetir la comparación con la misma población y las mismas particiones.

## 15. Reproducción

1. Obtener el CSV descrito en `data/README.md` y colocarlo en `data/raw/` o en la ruta que acepte el script.
2. Instalar `requirements.txt` en un entorno compatible. En esta revisión se comprobaron imports en Python 3.14.0; no se verificó instalación limpia.
3. Ejecutar notebooks 01, 04, 07–11 según las rutas y archivos guardados; 09g documenta la búsqueda de parámetros.
4. El análisis de errores requiere el CSV y `outputs/evaluacion_final/predicciones_test.csv.gz`; ejecuta `python -m src.analisis_errores` o corre las celdas de `12_analisis_errores.ipynb`.
5. No se debe confundir el script de baseline `python -m src.run_baseline` con la reproducción del flujo calibrado final; actualmente no existe un comando único `src.run_final.py` que reconstruya todas las etapas finales.

## Artefactos comprobados

- `outputs/optimizacion_final/configuracion_reproducible.json`
- `outputs/calibracion_probabilidades/metricas_promedio.csv`
- `outputs/evaluacion_final/metricas_finales_test.csv`
- `outputs/evaluacion_final/intervalos_bootstrap_95.csv`
- `outputs/analisis_errores/metricas_globales.csv`
- `reports/auditoria_integracion.md`
