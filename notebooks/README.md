# Notebooks oficiales


Los once notebooks tienen un **control de ejecución: consulta o reproducción** al inicio:

```python
REPRODUCIR = False
RUTA_DATOS = None
```

Con `False` se consulta la evidencia histórica disponible sin necesitar el CSV original, entrenar, buscar hiperparámetros, recalibrar ni escribir resultados.

Con `True`, reiniciar el kernel y ejecutar todas las celdas en orden. Se guardan métricas, figuras, `manifiesto.json` y `verificacion.csv` en una carpeta nueva de `outputs/reproducciones/`. El notebook 11 con `True` ejecuta el procedimiento completo, incluida la evaluación del test. El notebook 10 calibra con 2016 y el 11 evalúa 2017–2018.

## Cómo se verifica una reproducción frente a los resultados históricos

La **consulta** y la **reproducción** cumplen funciones distintas. En consulta, el notebook solo muestra resultados que ya estaban guardados: no vuelve a procesar los datos ni a entrenar el modelo, así que su estado queda como **No verificado**. En reproducción, ejecuta el procedimiento y guarda los resultados nuevos en una carpeta aparte; después compara esos archivos con la evidencia histórica disponible, sin reemplazarla.

La comparación revisa las tablas y archivos JSON que existan en ambas versiones. Para las tablas comprueba la cantidad de filas y columnas, y las alinea por campos identificadores disponibles —por ejemplo, modelo, año, configuración o métrica—. Si faltan esas claves o están repetidas y no se puede determinar qué fila corresponde a cuál, la comparación no fuerza una coincidencia: la marca como **No verificable**. En JSON se comparan los campos disponibles. Los números reales admiten una diferencia muy pequeña (`atol=1e-8`, `rtol=1e-6`); años, conteos, categorías y otros valores enteros deben coincidir exactamente. Los tiempos de ejecución se registran, pero no se espera que sean iguales entre equipos.

| Estado | Qué significa |
|---|---|
| **Coincide** | Se ejecutó el procedimiento y los valores comparables cumplen los criterios anteriores. |
| **Difiere** | Se ejecutó, pero al menos un tamaño, campo o valor comparable no coincide. |
| **No verificado** | Solo se consultaron resultados guardados; no se realizó una reproducción nueva. |
| **No verificable** | Falta un archivo histórico, no hay una correspondencia inequívoca entre filas o la evidencia disponible no permite hacer la comparación. |

Cada reproducción guarda un manifiesto.json con la configuración, las versiones de las herramientas y la información disponible sobre los datos utilizados. También guarda un verificacion.csv que indica si los resultados nuevos coinciden con los anteriores.

| Notebook | Procedimiento que se reproduce |
|---|---|
| 01_exploracion_inicial_baseline | Exploración, muestra histórica de 220,003 y variantes logísticas. |
| 02_arbol_decision | Muestra compartida, preparación del árbol y grilla de 168 configuraciones. |
| 03_gradient_boosting | Muestra compartida y grilla de 108 configuraciones; modo de prueba explícitamente incompleto. |
| 04_exploracion_completa | Carga compacta de 97 columnas, selección histórica de 26, análisis descriptivo y auditoría de carga. |
| 05_svm_muestra | Cuatro modelos con preprocesamiento compartido sobre la muestra inicial; SVM con márgenes sin calibrar; primera referencia real disponible en outputs/svm_muestra/. |
| 06_entrenamiento_busqueda_inicial | Modelos fijos y búsqueda aleatoria inicial, con cuatro cortes temporales. |
| 07_comparacion_variables_imputacion | Tres comparaciones separadas, con sus diferencias metodológicas. |
| 08_umbrales_sensibilidad | Umbrales y sensibilidad encadenados a los resultados nuevos. |
| 09_optimizacion_final | Sorteos originales, selección temporal al 50% y checkpoints nuevos. |
| 10_calibracion | Métodos y decisiones de calibración con desarrollo. |
| 11_evaluacion_final | Entrenamiento base, calibración, test, intervalos y errores; confirmación del test ya configurada; se activa con REPRODUCIR. |

Los notebooks pueden ejecutarse individualmente con sus entradas y configuraciones originales. El pipeline final `src/run_final.py` sigue siendo independiente. Consulta [la guía del repositorio](../README.md).
