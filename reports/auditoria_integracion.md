# Auditoría de integración del repositorio

Fecha de revisión: 13 de noviembre de 2026. Repositorio: `Josu31506/Proyecto-ML-Grupo8`.

## Estado Git y ramas

- `main` y `origin/main` estaban en `c4cdc7e` (`Agrega árbol de decisión...`).
- `gradient-boosting` estaba en `d2c49e4`, hijo directo de `main`; no había pull requests abiertos al momento de la consulta.
- El directorio local ya contenía cambios sin commit de notebooks, módulos, resultados y documentación. Se conservaron y se abrió una rama de trabajo; no se ejecutó reset ni limpieza destructiva.
- El CSV original está en la carpeta de trabajo local y no se debe subir. Está excluido por `.gitignore`.

## Inventario de artefactos relevantes

| Grupo/archivo | Estado local y remoto | Dependencias/observaciones | Decisión |
|---|---|---|---|
| `notebooks/01_exploracion_inicial.ipynb` | En `main` | Muestra de primera entrega; 220 mil préstamos, reglas antiguas de faltantes; algunas métricas históricas consultan test | Conservar como antecedente, no presentarlo como flujo final |
| `notebooks/02_avance_modelos.ipynb` | Solo local | Depende de `src/avance.py` y `outputs/avance/` | Conservar en la secuencia de avances |
| `notebooks/03_arbol_decision.ipynb` | En `main` | Árbol individual y comparación con logística de la primera etapa | Conservar como modelo exploratorio |
| `notebooks/03_auditoria_tipos_memoria.ipynb` | Solo local | Comparación de lectura y tipos; comparte prefijo `03` con el notebook del árbol, pero no es duplicado | Conservar y explicar su propósito; renombrar solo si se actualizan todos los enlaces |
| `notebooks/04_gradient_boosting.ipynb`, `src/boosting.py`, `src/run_boosting.py` | Solo en `gradient-boosting` | Reproduce una muestra pequeña y un corte simple; artefactos de métricas mezclan tamaños de muestra | Conservar como experimento histórico con advertencia de comparabilidad |
| `notebooks/04_exploracion_final.ipynb` | Solo local | Usa el archivo completo y revisa cobertura antes de los cortes | Conservar como exploración de datos completos |
| `notebooks/05_seleccion_variables.ipynb`, `06_arbol_30_variables.ipynb`, `Reduccion_dimensionalidad.ipynb` | Solo local | Experimentos con listas de 65/30/38; el notebook RFECV también calcula resultados de test | Conservar como exploratorios; marcar la exposición histórica del test |
| `notebooks/07_entrenamiento.ipynb`, `08_busqueda_hiperparametros.ipynb` | Solo local | Entrenamiento y búsqueda tempranos con 26 predictoras | Conservar como etapas históricas, no confundir con 09g |
| `notebooks/09*` | Solo local | Pruebas de umbral, faltantes, imputación y búsqueda final | Conservar como experimentos. La selección cambia entre 10 %, 50 % y 75 % según el experimento |
| `notebooks/10_calibracion_probabilidades.ipynb` | Solo local | Validación/calibración cronológica, cuatro años de validación | Usar para explicar el calibrador seleccionado |
| `notebooks/11_evaluacion_final.ipynb` | Solo local | Métricas finales en préstamos resueltos de 2017–2018 | Mantener como evaluación guardada, explicando las limitaciones de maduración y la exposición previa del test |
| `notebooks/12_analisis_errores.ipynb` | Añadido localmente | Calcula errores con umbral fijo 0.50 y segmenta por año, plazo, ingreso y solicitud | Conservar; no presenta el umbral como una regla de aprobación crediticia |
| `src/config.py`, `data.py`, `features.py`, `train.py`, `evaluate.py`, `experimentos.py`, `arbol.py`, `run_baseline.py`, `run_arbol.py` | Base disponible en GitHub; cambios locales en `config.py` y `features.py` | Código de primera etapa; `UMBRAL_FALTANTES=95` histórico y `UMBRAL_FALTANTES_FINAL=10` exploratorio | Conservar diferencias históricas explícitas |
| `src/modelado.py` | Solo local | Implementa cortes expansivos, selección por cobertura y modelos iniciales | Conservar para experimentos con 26 variables |
| `src/etapas_finales.py` | Solo local | Pipeline final; umbral `UMBRAL_FALTANTES=50`, filtros entrenados en cada base | Fuente efectiva de la calibración y evaluación final |
| `src/avance.py`, `benchmark_worker.py`, `benchmark_dtype_worker.py` | Solo local | Avance de todo el archivo y benchmarks de carga/memoria | Conservar; ejecutar benchmarks solo si se necesita reproducirlos |
| `outputs/evaluacion_final/` | Solo local | Contiene métricas, intervalos y predicciones comprimidas | Versionar tablas pequeñas seleccionadas; no subir `predicciones_test.csv.gz` salvo que se acuerde explícitamente |
| `outputs/calibracion_probabilidades/` y `outputs/optimizacion_final/` | Solo local | Resultados agregados reproducibles | Conservar tablas pequeñas clave; excluir predicciones por fila grandes |
| `reports/avance.md` y figuras de avance | Solo local | Material de presentación del avance | Conservar; revisar el texto para no mezclar resultados de muestra y de todos los datos |

## Hallazgos metodológicos priorizados

1. **Los resultados de la rama histórica de Gradient Boosting no son comparables directamente** con el pipeline final. La rama registra 1,668 observaciones de entrenamiento, 2,504 de validación y 830 de test (≈5 mil filas) y usa un único bloque temporal 2015–2016. El notebook declara 220,003 filas y no tiene celdas ejecutadas. Su JSON además mezcla esas métricas con las del árbol, que tiene 73,701/109,382/36,920 filas.
2. **El flujo final no usa 26 variables fijas.** En cada corte de optimización selecciona candidatas con ≤50 % de faltantes calculado en el conjunto base: 26 variables originales en 2013 y 60 en cada uno de los cortes 2014–2016. El promedio es 51.5; en el ajuste final se seleccionaron 60. No se debe resumir como un conjunto fijo de 26 o 59 variables.
3. **Los umbrales pertenecen a experimentos distintos.** La configuración histórica de 95 % está en `config.py`; el flujo de selección fija menor de 10 % hasta 2012 está en `modelado.py`; la optimización y la evaluación final aplican ≤50 % en cada tramo base desde `etapas_finales.py`. Los resultados finales se generaron con este último flujo y sin indicadores adicionales.
4. **El test fue consultado antes de cerrar el flujo final.** La primera entrega incluye métricas de prueba; el árbol y la reducción de dimensionalidad también reportan prueba. Las decisiones de calibración final se realizaron con validación y los resultados finales se generaron después, pero no es correcto afirmar que los datos de 2017+ nunca se inspeccionaron. Además, las etiquetas del archivo no contienen la fecha del desenlace.
5. **Maduración y población seleccionada.** En 2017–2018 hay 938,821 préstamos en total, pero solo 225,611 tienen los estados finales binarios elegibles. Excluir los activos significa que la muestra de test incluye préstamos cuyo desenlace se observó; no representa todo el originado en esos años.
6. **La calibración temporal no es una reconstrucción de disponibilidad histórica completa.** El CSV presenta estados finales, pero no la fecha en que cada préstamo alcanzó el estado. Los cortes por `issue_d` separan años, aunque no prueban que las etiquetas del entrenamiento hubieran sido conocidas en ese momento.
7. **`HistGradientBoosting` es el estimador final de boosting.** En los artefactos aparece con clave `gradient_boosting`; conviene aclarar el nombre del algoritmo para no confundirlo con el Gradient Boosting histórico de la otra rama.
8. **No se guardó el modelo serializado final.** Se guardaron los parámetros, las probabilidades y métricas. Por eso el notebook de errores compara las predicciones guardadas; la importancia de variables histórica no se presenta como explicación del modelo final.

## Dependencias y ejecución

El entorno actual ejecutó Python 3.14.0, pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.8.0 y matplotlib 3.10.8. `requirements.txt` fija paquetes, pero no declara la versión exacta de Python y no se hizo una instalación limpia de entorno nuevo. Por tanto, se verificaron imports y pruebas locales; no se afirma que `pip install -r requirements.txt` se haya probado en un entorno vacío.

Los notebooks requieren ejecutarse desde el repositorio o desde `notebooks/` y esperan encontrar el dataset en las rutas descritas en `data/README.md` o `src/config.py`. Evítese guardar rutas personales.

## Comparación numérica de Gradient Boosting

| Experimento | Validación | Test | Lectura |
|---|---:|---:|---|
| Rama `gradient-boosting`, sin variables LC | ROC-AUC 0.7007 / AP 0.3768, n=2,504 | ROC-AUC 0.6993 / AP 0.3978, n=830 | Experimento exploratorio pequeño; no se debe comparar en igualdad con el pipeline final |
| Rama `gradient-boosting`, con variables LC | ROC-AUC 0.7157 / AP 0.3955, n=2,504 | ROC-AUC 0.7115 / AP 0.4097, n=830 | Añade una lista distinta de variables; no es una comparación aislada |
| Nuestro HistGradientBoosting optimizado, promedio cuatro cortes | ROC-AUC 0.7105 / AP 0.3695; media de 51.5 columnas originales | — | Entrenamiento completo, mediana/categorías ajustadas en cada base; métricas no calibradas de búsqueda |
| Nuestro HistGradientBoosting calibrado | Validación por diseño de calibración: ROC-AUC 0.6974 / AP 0.3520 | ROC-AUC 0.7032 / AP 0.3779, n=225,611 | Artefacto final guardado, probabilidades sigmoides; no comparar sin atender a los cortes y la muestra |

Las diferencias en tamaño, años, selección de variables, preprocesamiento y calibración impiden decir que la rama de boosting “ganó” o “perdió” frente al nuestro. Su resultado sirve para documentar un experimento inicial.
