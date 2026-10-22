# Avance posterior a la primera entrega

Resultados ejecutados el 21 de octubre de 2026. Fuente: `outputs/avance/<modo>/metrics.json`.

## Qué pidió el profesor y qué hicimos

El anuncio pide explicar los avances, mostrar nuevos hallazgos y comparar la línea base con otro modelo del curso. Se permite una muestra para la revisión y se recomienda llegar al dataset completo. Los ejemplos de clasificación multiclase son contenido de clase, no una instrucción para cambiar el objetivo del proyecto.

1. Conservamos la primera entrega y agregamos `notebooks/02_avance_modelos.ipynb`.
2. Implementamos carga por bloques con dos modos: muestra y todos los elegibles.
3. Analizamos qué porcentaje de cada año y plazo conserva las etiquetas elegidas.
4. Comparamos una nueva familia, árbol de decisión, con la regresión logística sin indicadores ni LC sobre las mismas entradas.
5. Reservamos prueba y no calculamos nuevas métricas sobre ella en este avance.

## Alcance de los datos

El archivo tiene 2 260 701 filas. Hay 1 345 310 préstamos con las etiquetas exactas Fully Paid o Charged Off. El modo completo carga todos ellos: 451 059 para entrenamiento, 668 640 para validación y 225 611 reservados para prueba.

Usar todos los elegibles no significa entrenar con todas las filas ni incluir préstamos activos sin resultado. Se mantienen las reglas de exclusión de información posterior al préstamo y los cortes provisionales de la primera entrega. Las columnas excluidas por faltantes se deciden solo con entrenamiento.

## Comparación ejecutada

| Modo | Modelo | Entrenamiento | Validación | ROC-AUC validación | AP validación | Segundos de entrenamiento |
|---|---|---:|---:|---:|---:|---:|
| muestra | logistica_sin_indicadores | 73,701 | 109,382 | 0.7121 | 0.4016 | 1.57 |
| muestra | arbol_decision | 73,701 | 109,382 | 0.6717 | 0.3515 | 1.60 |
| completo | logistica_sin_indicadores | 451,059 | 668,640 | 0.7156 | 0.4093 | 6.74 |
| completo | arbol_decision | 451,059 | 668,640 | 0.6797 | 0.3676 | 13.86 |

El árbol usa profundidad máxima 6 y al menos 0.5 % de las filas de entrenamiento por hoja. Es una configuración inicial fija, no una búsqueda de parámetros. La logística conserva sus parámetros anteriores. El umbral de cada modelo maximiza F1 en validación.

En ambos modos la logística supera al árbol en ROC-AUC y AP. No se cambia de modelo solo por usar una familia nueva. El baseline sobre la muestra reproduce las métricas de la primera entrega a cuatro decimales.

Los tiempos miden solo `fit`, no lectura, preparación ni evaluación. Son orientativos para este equipo. Muestra y completo usan distintas filas de validación: la diferencia entre modos no permite atribuir una mejora únicamente a más entrenamiento. La comparación entre modelos dentro de cada modo sí utiliza las mismas filas y variables.

## Nuevo hallazgo: cobertura por año y plazo

Entre los préstamos de 2013 a 60 meses, 99.97 % tiene las etiquetas elegidas; para 2018 a 60 meses la proporción es 9.99 %. En 2016 a 60 meses, 54.77 % es elegible y, dentro de ese grupo, 36.43 % terminó en Charged Off.

Estas cifras describen la selección del dataset, no la probabilidad de incumplimiento de todos los préstamos otorgados. Los demás estados incluyen activos, atrasos y categorías especiales de política crediticia. Por eso el porcentaje elegible no equivale exactamente al porcentaje que terminó de pagarse ni mide por sí solo el seguimiento. En los años iniciales también influyen los estados especiales excluidos.

No inferimos la fecha de actualización a partir del nombre del archivo. Un préstamo a 60 meses puede pagarse anticipadamente. Queda pendiente verificar esa fecha y si los resultados de entrenamiento ya se conocían cuando se simula predecir préstamos posteriores.

## Guion para la exposición

1. Mostrar la tabla de conteos: de aproximadamente 220 mil a todos los elegibles.
2. Explicar que la lectura por bloques y la selección de columnas reducen memoria.
3. Mostrar el gráfico por año y plazo y explicar la cobertura desigual.
4. Presentar el árbol y comparar sus métricas con la logística.
5. Concluir que esta configuración no mejora el baseline y explicar el siguiente paso: verificar seguimiento y después ampliar o ajustar modelos usando validación.

## Reproducción

```bash
python -m src.avance --modo muestra
python -m src.avance --modo completo
python -m unittest discover -s tests -v
```

El notebook utiliza estas mismas funciones; carga los resultados guardados y los regenera si faltan. Para forzar ambas corridas, establecer `RECALCULAR = True`. Las cuatro pruebas comprueban elegibilidad y exclusiones, reproducibilidad de muestra, independencia del preprocesamiento respecto a validación, ausencia de evaluación de prueba y detección de fechas inválidas.
