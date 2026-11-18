# Predicción de incumplimiento de préstamos — Lending Club

Proyecto final del curso **Machine Learning**, **UTEC**, **Grupo 8**. Estimamos el riesgo individual de que un préstamo personal aceptado termine declarado incobrable. El resultado sirve de apoyo al análisis crediticio.

**Integrantes:** Gavino Navarrete Adrian Piero; Hernández Yataco, Josué Renzo; Rojas Cruz, José Alvaro; Sanchez Dominguez, Matias Sebastian.

**Pregunta predictiva del proyecto:** Dado un préstamo que Lending Club acaba de aprobar, y usando únicamente la información disponible el día del desembolso, ¿qué probabilidad hay de que termine declarado incobrable (`Charged Off`) en lugar de pagarse por completo (`Fully Paid`)?

Cada fila representa un préstamo. La variable objetivo es `Fully Paid = 0` y `Charged Off = 1`. El modelo principal es **HistGradientBoosting**, comparado con regresión logística y SVM lineal en el flujo final. El árbol de decisión forma parte de los experimentos iniciales.

## Estado de la entrega

- Hay **diez notebooks oficiales** y código reutilizable para reproducir sus experimentos.
- `src/run_final.py` permite **entrenar, calibrar, guardar, evaluar y predecir** sin ejecutar todos los notebooks.
- Los tres artefactos locales de `models/final_01/` y su manifiesto están presentes. Registran entrenamiento con **826,604 préstamos hasta 2015** y calibración con **293,095 de 2016**. Los modelos no se versionan; deben regenerarse.
- **La evaluación real de los tres modelos serializados reprodujo las métricas históricas del notebook 10**, a los seis decimales mostrados, sobre 225,611 préstamos de 2017–2018. Los CSV locales de evaluación se contrastaron con la referencia histórica. Las 19 pruebas automatizadas también finalizaron correctamente según la consola local aportada por el usuario; no son resultados de GitHub Actions.
- También existe una comparación inicial ejecutada sobre **220,003 préstamos** en `outputs/gradient_boosting_220000/`. Es un experimento distinto del flujo final calibrado sobre desarrollo completo.

## Dataset y población del análisis

Se utiliza [Lending Club Loan Data, de wordsforthewise](https://www.kaggle.com/datasets/wordsforthewise/lending-club), archivo `accepted_2007_to_2018Q4.csv.gz` o su CSV descomprimido. La auditoría inicial encontró **2,260,701 filas, 151 columnas y 1,345,310 préstamos** con las etiquetas binarias seleccionadas. Los datos fuente no se incluyen en GitHub.

Los préstamos `Current`, atrasados y otros estados no se convierten artificialmente en incumplimientos: quedan fuera de la población binaria. Los resultados describen préstamos aceptados con desenlace final observado; no describen cómo habrían pagado los rechazados.

Instrucciones de obtención: [data/README.md](data/README.md). El [diccionario en español](<data/Lending Club Data Dictionary Approved - ES.csv>) ayuda a interpretar las variables originales.

## Estructura del repositorio

```text
Proyecto-ML-Grupo8/
├── .gitignore                      # Exclusiones de datos, modelos y temporales
├── README.md                       # Guía del proyecto y reproducción
├── proposal.md                     # Propuesta de la primera entrega
├── requirements.txt                # Versiones de dependencias
├── accepted_2007_to_2018Q4.csv       # Archivo fuente local, ignorado por Git
├── data/
│   ├── README.md                   # Cómo obtener el dataset
│   └── Lending Club Data Dictionary Approved - ES.csv
├── notebooks/                      # Diez etapas oficiales de investigación
│   └── README.md                   # Orientación de lectura
├── src/                            # Preparación, experimentos y pipeline reutilizable
├── tests/
│   ├── test_integracion.py         # Consistencia de datos y preparación
│   └── test_pipeline_final.py      # Pipeline con datos sintéticos
├── outputs/                        # Métricas, configuraciones y tablas por experimento
│   └── README.md                   # Guía de los resultados conservados
├── reports/
│   ├── informe_final.md            # Metodología, resultados y limitaciones
│   ├── presentacion_esquema.md     # Guion de exposición
│   └── figures/                    # Figuras de baseline, árbol, boosting y errores
└── models/
    ├── .gitkeep                    # Conserva la carpeta en GitHub
    └── final_01/                   # Solo local: tres joblib y manifest.json
```

El árbol distingue archivos públicos de datos y artefactos locales ignorados. `notebooks/` documenta la investigación; `src/` implementa los procedimientos; `outputs/` conserva resultados; `reports/` los comunica; `models/` guarda objetos entrenados; `tests/` verifica su comportamiento.

## Diez notebooks oficiales

El orden sigue las decisiones del proyecto. Para usar el modelo final basta con el pipeline de más abajo; no es necesario ejecutar todos los experimentos.

| Nº | Archivo | Objetivo y experimentos | Decisión o resultado principal | Siguiente etapa |
|---:|---|---|---|---|
| 01 | [01_exploracion_inicial_baseline.ipynb](notebooks/01_exploracion_inicial_baseline.ipynb) | EDA, muestra, faltantes, outliers y regresiones logísticas iniciales | Muestra de 220,003; referencia sin indicadores y escenarios alternativos | Comparar relaciones no lineales |
| 02 | [02_arbol_decision.ipynb](notebooks/02_arbol_decision.ipynb) | Árbol libre/podado y grilla de 168 configuraciones | Comparación con logística usando los mismos periodos de la muestra | Comparar con un ensamble |
| 03 | [03_gradient_boosting.ipynb](notebooks/03_gradient_boosting.ipynb) | Grilla de 108 configuraciones de boosting; recalcula logística y árbol sobre la muestra verificada | Ejecución de 220,003 guardada: ROC-AUC de validación de boosting 0.71556 | Ampliar a todos los préstamos elegibles |
| 04 | [04_exploracion_completa.ipynb](notebooks/04_exploracion_completa.ipynb) | 97 columnas candidatas/auxiliares, tipos compactos, cobertura, correlaciones e imputación descriptiva | Primera selección fija de 26 variables con menos de 10 % de faltantes hasta 2012; anexo de carga/memoria | Entrenar con validación temporal |
| 05 | [05_entrenamiento_busqueda_inicial.ipynb](notebooks/05_entrenamiento_busqueda_inicial.ipynb) | Modelos iniciales y primera búsqueda aleatoria con 26 variables | Cuatro cortes expansivos; referencia para las comparaciones posteriores | Revisar selección e imputación |
| 06 | [06_comparacion_variables_imputacion.ipynb](notebooks/06_comparacion_variables_imputacion.ipynb) | 26 variables, 59 exploratorias, selección por corte; mediana, indicadores y NaN nativo | Los indicadores no muestran una ventaja importante y consistente; las 59 fijas usan cobertura hasta 2016 y son exploratorias | Revisar el umbral de faltantes |
| 07 | [07_umbrales_sensibilidad.ipynb](notebooks/07_umbrales_sensibilidad.ipynb) | Umbrales 10/30/50/75 %; disponibilidad y sensibilidad 50 frente a 75 | Se adopta 50 % por cautela temporal. El 75 % tuvo mayor promedio, pero su ventaja ROC de boosting se concentró en un año | Optimizar con la regla elegida |
| 08 | [08_optimizacion_final.ipynb](notebooks/08_optimizacion_final.ipynb) | Configuraciones aleatorias y selección de variables dentro de cada entrenamiento | Fija parámetros de logística, SVM e HistGradientBoosting; no pretende un óptimo universal | Evaluar probabilidades |
| 09 | [09_calibracion.ipynb](notebooks/09_calibracion.ipynb) | Original, sigmoide e isotónica con base/calibración/evaluación cronológicas | Se elige sigmoide para los tres modelos usando desarrollo | Evaluar las configuraciones fijadas |
| 10 | [10_evaluacion_final.ipynb](notebooks/10_evaluacion_final.ipynb) | Test histórico, intervalos, curvas y errores por segmentos | Comparación final calibrada e interpretación de sus límites | Inferencia mediante artefactos guardados |

La muestra inicial usa entrenamiento **2007–2014**, validación **2015–2016** y prueba **2017–2018**. El flujo posterior utiliza todos los préstamos elegibles y reglas ajustadas por periodo. Sus tamaños y puntuaciones no deben mezclarse como si pertenecieran al mismo experimento.

## Código reutilizable de `src/`

| Módulo | Entrada y responsabilidad | Salida y consumidores |
|---|---|---|
| [config.py](src/config.py) | Semilla, rutas, etiquetas, cortes y constantes de escenarios | Configuración compartida; los umbrales del experimento inicial y del flujo final son distintos |
| [data.py](src/data.py) | CSV original; auditoría, normalización, muestra proporcional por bloques y partición anual | Muestra y auditoría para baseline, árbol y boosting inicial |
| [features.py](src/features.py) | Tablas de préstamos; excluye auxiliares, identificadores, texto y variables posteriores; prepara fechas y cobertura | Predictoras y preprocesadores usados por los escenarios y etapas finales |
| [train.py](src/train.py) | Preprocesador y tipo de referencia | Constructores de modelos iniciales y tabla de coeficientes; auxiliar del baseline |
| [evaluate.py](src/evaluate.py) | Etiquetas y puntuaciones | Métricas, umbral elegido en validación y exportación JSON de experimentos iniciales |
| [experimentos.py](src/experimentos.py) | Particiones temporales y escenarios logísticos | Matrices, modelos y métricas comparables de baseline; usado por notebooks y ejecutores iniciales |
| [arbol.py](src/arbol.py) | Particiones y grilla del árbol | Preparación, búsqueda, modelo, métricas e importancias; usado por notebook 02 y `run_arbol` |
| [boosting.py](src/boosting.py) | Particiones y grilla de HistGradientBoosting inicial | Preparación, búsqueda, modelo e importancia por permutación; usado por `run_boosting` |
| [comparabilidad_muestra.py](src/comparabilidad_muestra.py) | CSV, muestra y referencias iniciales | Reconstrucción de IDs/orden y huellas de particiones; protege la comparación inicial |
| [modelado.py](src/modelado.py) | Dataset, periodos y modelos candidatos | Tipos auditados, cortes, imputación y métricas de desarrollo; usado por comparaciones y búsquedas |
| [etapas_finales.py](src/etapas_finales.py) | Dataset o matrices, parámetros congelados y periodos | Selección al 50 %, preparación, modelos, calibradores, métricas y `PipelineFinal`; usado por notebooks 09–10 y ejecutor final |
| [run_final.py](src/run_final.py) | Argumentos de entrenamiento, evaluación o predicción | Artefactos completos, evaluaciones independientes de ejecución o probabilidades de préstamos nuevos |
| [analisis_errores.py](src/analisis_errores.py) | Probabilidades, etiquetas y segmentos del CSV | Verificación de alineación y métricas globales/por segmento; usado por notebook 10 y evaluación del pipeline |
| [rutas.py](src/rutas.py) | Ubicación del proyecto o variable `PROYECTO_ML_RAIZ` | Resolución común de la raíz para notebooks y scripts |
| [reproducir.py](src/reproducir.py) | Nombre de experimento y destino nuevo | Despacha comparaciones a carpetas nuevas bajo `outputs/reproducciones/` |
| [preparacion_temporal.py](src/preparacion_temporal.py) | Predictoras y fecha de otorgamiento | Transformación compartida de fechas y tipos para las comparaciones |
| [entrenamiento_inicial.py](src/entrenamiento_inicial.py) | Dataset y carpeta de salida | Referencias con 26 variables; regeneración desde notebook 05 |
| [comparacion_variables.py](src/comparacion_variables.py) | Dataset y reglas de selección | Métricas 26/59/selección temporal, con/sin indicadores; notebook 06 |
| [comparacion_imputacion.py](src/comparacion_imputacion.py) | Dataset y tratamientos de faltantes | Comparación mediana/indicadores/NaN nativo; notebook 06 |
| [comparacion_lineales.py](src/comparacion_lineales.py) | Dataset y configuraciones lineales ya elegidas | Comparaciones de logística/SVM por variables y tratamiento; notebook 06 |
| [comparacion_umbrales.py](src/comparacion_umbrales.py) | Dataset y cuatro umbrales fijos | Selección por entrenamiento y métricas por modelo/año; notebook 07 |
| [sensibilidad_faltantes.py](src/sensibilidad_faltantes.py) | Dataset y tablas de comparación de umbrales | Cobertura, diferencias anuales e intervalos 50/75; notebook 07 |
| [auditoria_memoria.py](src/auditoria_memoria.py) | CSV completo y destino | Comprobación de tipos sin pérdida y comparación de carga; notebook 04 |
| [benchmark_worker.py](src/benchmark_worker.py) | Configuración de lectura, método y destino | Medición de carga directa/bloques y huella de igualdad; auxiliar de auditoría |
| [run_baseline.py](src/run_baseline.py) | CSV y objetivo de muestra | Ejecuta el baseline inicial y genera métricas/figuras |
| [run_arbol.py](src/run_arbol.py) | CSV y objetivo de muestra | Ejecuta búsqueda y evaluación del árbol inicial |
| [run_boosting.py](src/run_boosting.py) | CSV, muestra 220000, modo y destinos vacíos | Verifica la muestra, recalcula referencias y ejecuta la grilla inicial sin sobrescribir resultados anteriores |

## Metodología

```text
Dataset original → filtrado y limpieza → objetivo binario → control de leakage
→ separación temporal → selección de variables → preprocesamiento
→ entrenamiento → calibración → evaluación → predicción
```

Se excluyen pagos, recuperaciones, saldos del préstamo posteriores, FICO actualizado, planes de dificultades y acuerdos de deuda. El flujo final también excluye el bloque de precio/calificación de Lending Club. La disponibilidad exacta al otorgamiento debe confirmarse antes de cualquier aplicación real.

En la etapa de comparación se usaron cuatro cortes expansivos:

| Entrenamiento | Validación |
|---|---|
| 2007–2012 | 2013 |
| 2007–2013 | 2014 |
| 2007–2014 | 2015 |
| 2007–2015 | 2016 |

Las combinaciones aleatorias corresponden a **hiperparámetros**; las observaciones nunca se mezclan aleatoriamente entre años. Para cada corte, la cobertura de variables, las medianas, categorías y escalas se aprenden solo con entrenamiento.

En el flujo final se conservan variables con **50 % o menos de faltantes** y algún valor observado en la base. Se imputan numéricas con mediana y categóricas con «Sin información», se usa One-Hot con manejo de categorías nuevas y se aplica el escalado de la implementación histórica. No se añaden indicadores de faltantes. La configuración final histórica retuvo **60 variables originales y 136 columnas transformadas**; se guarda la selección ajustada y su orden, sin calcularlos con información futura.

La base final es **2007–2015**, la calibración es **2016** y la evaluación histórica es **2017–2018**. El estimador y el preprocesador permanecen congelados durante la calibración sigmoide. Se aplica logit a las probabilidades de logística/boosting y se usa el margen de `LinearSVC`; después se ajusta la sigmoide sobre 2016. El estimador no se vuelve a entrenar incorporando ese año.

## Resultados históricos finales

Valores verificados en [metricas_finales_test.csv](outputs/evaluacion_final/metricas_finales_test.csv):

| Modelo calibrado | ROC-AUC | PR-AUC (AP) | Brier | Log Loss |
|---|---:|---:|---:|---:|
| Regresión logística | 0.69259 | 0.36039 | 0.15591 | 0.48501 |
| SVM lineal | 0.69178 | 0.35988 | 0.15597 | 0.48680 |
| **HistGradientBoosting** | **0.70321** | **0.37791** | **0.15364** | **0.47428** |

HistGradientBoosting es el principal por su mejor discriminación y menor error probabilístico entre estos tres candidatos en la comparación documentada. ROC-AUC y AP se calculan con puntuaciones continuas; Brier y Log Loss evalúan probabilidades. Mayor ROC-AUC/AP y menor Brier/Log Loss son preferibles. Esto no implica certeza individual ni un óptimo universal.

El test histórico incluye **225,611 préstamos**, con **21.28 % Charged Off**. Del total de 938,821 préstamos emitidos en 2017–2018 en el archivo, muchos seguían activos. Las métricas se refieren solo a la subpoblación con resultado final.

La evaluación local posterior del pipeline guardado **confirmó la reproducción de estas métricas**. La sección siguiente distingue la evidencia real de test de las pruebas sintéticas; esta actualización documental no repite ninguna ejecución.

### Validación final del pipeline reproducible

El entrenamiento con `python -m src.run_final --modo entrenar --salida models/final_01` finalizó correctamente y generó `regresion_logistica.joblib`, `svm_lineal.joblib`, `gradient_boosting.joblib` y `manifest.json`. Estos artefactos permanecen exclusivamente locales y pueden regenerarse con el código y los datos originales; no están disponibles en GitHub.

Las pruebas ejecutadas con `python -m unittest discover -s tests -v` finalizaron con la salida local proporcionada por el usuario:

```text
Ran 19 tests in 6.795s

OK
```

Son pruebas con datos sintéticos de particiones, prevención de leakage, preprocesamiento, predicción, calibración, serialización y evaluación explícita. La salida de consola acredita la ejecución local reportada; no se presenta como un registro de GitHub Actions ni sustituye la evaluación real de Lending Club.

Posteriormente se ejecutó la evaluación explícita de los modelos guardados:

```powershell
python -m src.run_final --modo evaluar --modelos models/final_01 --datos accepted_2007_to_2018Q4.csv --conjunto test --confirmar-test
```

| Modelo calibrado con sigmoide | ROC-AUC | PR-AUC (AP) | Brier Score | Log Loss | n |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.692591 | 0.360386 | 0.155911 | 0.485014 | 225611 |
| SVM lineal | 0.691777 | 0.359876 | 0.155972 | 0.486804 | 225611 |
| HistGradientBoosting | 0.703213 | 0.377905 | 0.153642 | 0.474278 | 225611 |

El test corresponde a 2017–2018 y su prevalencia de incumplimiento es 0.212822. La evidencia persistente local se encuentra en `outputs/pipeline_final/test_20261117_195412_266845/metricas.csv` y `alcance.json`, excluidos de Git junto con sus predicciones individuales. Se compararon las cuatro métricas, la prevalencia y n de los tres modelos con [la referencia histórica](outputs/evaluacion_final/metricas_finales_test.csv): los 18 campos numéricos coinciden en los CSV y, por tanto, a los seis decimales presentados. No se recalcularon predicciones ni se repitió el test durante esta revisión documental.

Esto confirma la reproducción numérica de las métricas mediante los artefactos serializados, no una validación externa ni la identidad fila a fila de todas las predicciones. El test ya había sido consultado durante el desarrollo histórico; siguen vigentes la selección de préstamos con desenlace observado, las limitaciones de maduración de etiquetas y la ausencia de evaluación prospectiva estrictamente independiente.

### Comparación inicial con muestra

La ejecución guardada en [gradient_boosting_220000/metrics_boosting.json](outputs/gradient_boosting_220000/metrics_boosting.json) recalculó los tres modelos sobre 220,003 observaciones:

| Modelo | ROC-AUC validación | AP validación | ROC-AUC prueba |
|---|---:|---:|---:|
| Logística sin indicadores | 0.71205 | 0.40161 | 0.69019 |
| Árbol con parámetros fijados | 0.67448 | 0.35482 | 0.63458 |
| HistGradientBoosting inicial | 0.71556 | 0.40716 | 0.69135 |

Estos valores pertenecen a la muestra y no sustituyen la evaluación final calibrada. La referencia anterior de Boosting con objetivo 5,000 y 5,002 efectivos permanece en `outputs/metrics_boosting.json`, marcada como una ejecución de tamaño distinto. El [manifiesto de muestra](outputs/gradient_boosting_220000/muestra_verificada.json) registra las comprobaciones de IDs y particiones.

## Instalación y datos

Ejecutar desde la raíz del repositorio, por ejemplo en PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

El entrenamiento local guardado registra Python 3.14.0, pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.8.0 y joblib 1.5.3. Las pruebas de implementación también se ejecutaron con Python 3.12. El cargador verifica que las versiones numéricas y de serialización coincidan con las registradas en el artefacto.

Descargar el CSV desde la fuente indicada en [data/README.md](data/README.md) y colocarlo en `data/raw/`, `data/`, la raíz, o pasar `--datos RUTA`. `src.config.localizar_datos` también revisa Downloads. El CSV local descomprimido ocupa aproximadamente 1.6 GB; algunas etapas necesitan memoria adicional para matrices y copias. Las carpetas de datos y modelos se mantienen fuera de Git.

## Pipeline final reproducible

El pipeline reutiliza la configuración definitiva; **no realiza búsquedas de hiperparámetros**. No requiere ejecutar todos los notebooks.

### Entrenar y calibrar

```powershell
python -m src.run_final --modo entrenar --salida models/final_01
```

Se puede añadir `--datos accepted_2007_to_2018Q4.csv`. Este modo solo utiliza base 2007–2015 y calibración 2016; no evalúa automáticamente el test. Genera:

- `regresion_logistica.joblib`, `svm_lineal.joblib` y `gradient_boosting.joblib`.
- `manifest.json`: hashes, modelo principal, selección/orden de variables, esquema de entrada, parámetros, periodos y versiones.

Cada joblib contiene selección, preprocesador, estimador y calibrador. La carpeta `models/final_01/` ya existe **localmente** y el comando rechazará sobrescribirla. Para una reproducción nueva, usar `--salida models/final_02` o no indicar `--salida`, que crea una carpeta fechada. Los artefactos no se publican en GitHub.

### Evaluar artefactos guardados

```powershell
python -m src.run_final --modo evaluar --modelos models/final_01 --datos accepted_2007_to_2018Q4.csv --conjunto validacion
```

`validacion` corresponde a **2016, usado para ajustar la sigmoide**. Es una revisión descriptiva del periodo de calibración, no una evaluación independiente.

El test histórico requiere autorización explícita en el comando:

```powershell
python -m src.run_final --modo evaluar --modelos models/final_01 --datos accepted_2007_to_2018Q4.csv --conjunto test --confirmar-test
```

El modo carga artefactos sin reajustarlos y escribe una ejecución nueva bajo `outputs/pipeline_final/`, o en el destino dado por `--salida`. Produce métricas, predicciones, clasificación con umbral 0.5, errores por segmentos, curvas, `costos_por_umbral.csv` y un registro de alcance. Ese archivo compara costos configurables mediante `src.analisis_errores`; sus valores predeterminados son 1 por falso positivo y 5 por falso negativo. El umbral es descriptivo; no es un umbral de negocio optimizado. El test no debe reutilizarse para elegir modelos, variables, calibración ni parámetros.

### Predecir préstamos nuevos

```powershell
python -m src.run_final --modo predecir --modelos models/final_01 --datos nuevos_prestamos.csv --modelo-prediccion svm_lineal
```

El argumento `--modelo-prediccion` permite elegir `gradient_boosting` (por defecto), `svm_lineal` o `regresion_logistica`. El CSV debe incluir las `columnas_entrada` del modelo elegido con sus nombres originales. Las columnas deben existir, aunque sus valores pueden faltar. No se exigen `loan_status`, `default`, pagos ni variables posteriores. Cuando se usa antigüedad crediticia, las fechas siguen el formato Lending Club, por ejemplo `Jan-2016`. Las columnas adicionales se ignoran.

Se carga HistGradientBoosting con sus transformaciones congeladas, se procesa el CSV por bloques y se guarda `fila_entrada` y `probabilidad_charged_off` en un CSV nuevo bajo `outputs/predicciones/`. `--salida outputs/predicciones_nuevas_01.csv` permite elegir el archivo. La inferencia no vuelve a entrenar ni necesita el dataset original.

### Parámetros de la interfaz

| Argumento | Uso |
|---|---|
| `--modo` | `entrenar`, `evaluar` o `predecir` |
| `--datos` | Ruta del CSV; obligatoria en evaluación y predicción |
| `--modelos` | Carpeta con manifiesto y joblib; obligatoria en evaluación y predicción |
| `--salida` | Carpeta nueva para entrenamiento/evaluación; archivo CSV nuevo para predicción |
| `--conjunto` | `validacion` (2016) o `test` (2017–2018) |
| `--confirmar-test` | Habilita explícitamente la evaluación del test histórico |
| `--modelo-prediccion` | Modelo serializado para `predecir`; por defecto `gradient_boosting` |

Consultar `python -m src.run_final --help`. Los destinos existentes no se sobrescriben. Una ejecución interrumpida debe repetirse en un destino nuevo.

## Reproducir experimentos de investigación

Los notebooks 04–07 incluyen controles `REGENERAR_*`, desactivados por defecto, para ejecutar solo la comparación deseada. `src/reproducir.py` guarda sus salidas en una carpeta nueva. En el notebook 07 se puede indicar una carpeta de tablas de umbrales regeneradas para el análisis posterior de sensibilidad.

La comparación inicial de Boosting puede repetirse con:

```powershell
python -m src.run_boosting --muestra 220000 --salida outputs/gb_220000_repeticion2 --figuras reports/figures/gb_220000_repeticion2
```

La grilla completa tiene 108 combinaciones; `--modo prueba` usa una combinación conservando las observaciones y destinos distintos. Los resultados actuales de 220 mil ya existen, por lo que el destino predeterminado no puede sobrescribirse. Baseline y árbol inicial tienen ejecutores `python -m src.run_baseline` y `python -m src.run_arbol`; estos ejecutores históricos escriben sus rutas originales, por lo que deben usarse de forma consciente si se quieren regenerar esas referencias.

## Pruebas automatizadas

```powershell
python -m unittest discover -s tests -v
```

Las pruebas usan datos sintéticos: particiones temporales, exclusión de leakage, selección e imputación solo en base, columnas/categorías nuevas, inferencia sin objetivo, guardado/carga, rangos de probabilidad, semilla, protección contra sobrescritura y bloqueo del test. También comparan el pipeline con la implementación histórica sobre datos sintéticos. No entrenan ni evalúan Lending Club.

## Resultados, informe y publicación

[outputs/README.md](outputs/README.md) describe métricas y configuraciones por etapa. El [informe final](reports/informe_final.md) desarrolla resultados y límites; el [guion de exposición](reports/presentacion_esquema.md) organiza la presentación. Las figuras están en [reports/figures/](reports/figures/) y en las carpetas de resultados correspondientes.

`.gitignore` excluye el dataset, los joblib y manifiestos locales de modelos, entradas de préstamos nuevos, predicciones individuales grandes, entornos, cachés, logs y temporales. Los **CSV y JSON agregados pequeños sí se incluyen** para que puedan verificarse los resultados publicados. Las probabilidades individuales que algunas celdas usan para volver a graficar o segmentar se regeneran localmente y no se descargan con el repositorio.

La rama de integración es `docs/entrega-final`. Las etapas históricas están publicadas como ramas acumulativas y sus commits originales ya forman parte del historial de la entrega. `gradient-boosting` se conserva como contribución independiente: su implementación inicial se incorporó al flujo consolidado sin restaurar un notebook duplicado. `main` se mantiene separado hasta revisar la integración.

## Evolución del proyecto por ramas

Las ramas representan **versiones históricas acumulativas** del repositorio. Cada una conserva los notebooks, módulos Python y outputs que existían en su commit, incluidos experimentos que posteriormente se descartaron o consolidaron. No son carpetas aisladas por tema ni variantes que deban mezclarse para ejecutar el modelo final.

Los **diez notebooks oficiales**, el pipeline reutilizable y la documentación vigente están consolidados en `docs/entrega-final`. Las ramas históricas permiten revisar cómo se llegó a esa versión; sus nombres de notebooks y criterios de preparación corresponden a su momento de desarrollo.

| Rama | Etapa y commit de referencia | Experimentos principales | Notebooks más relevantes en esa rama | Funcionalidades Python incorporadas o utilizadas | Relación con otras etapas |
|---|---|---|---|---|---|
| [feat/exploracion-inicial](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/feat/exploracion-inicial) | Primera entrega completa; `b4901e3` | Exploración y limpieza inicial, muestra de 220,003, regresiones logísticas y primeras métricas | `01_exploracion_inicial.ipynb` (1 notebook) | `config.py`, `data.py`, `features.py`, `train.py`, `evaluate.py`, `experimentos.py`, `run_baseline.py`: carga, preparación y baseline | Apunta directamente al commit original anterior al árbol; antecede a `main` y a las etapas posteriores |
| [main](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/main) | Primera comparación con árbol; `c4cdc7e` | Baseline y árbol, búsqueda de 168 configuraciones e interpretación | `01_exploracion_inicial.ipynb`, `03_arbol_decision.ipynb` (2 notebooks) | Añade `arbol.py` y `run_arbol.py`, reutilizando la preparación inicial | Punto de partida común de la contribución de Boosting y del desarrollo con todos los datos; no se actualiza con esta publicación |
| [gradient-boosting](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/gradient-boosting) | Contribución inicial independiente; `d2c49e4` | HistGradientBoosting y búsqueda en grilla; sus outputs preliminares no equivalen a la comparación final de muestra completa | `04_gradient_boosting.ipynb`, además de baseline y árbol (3 notebooks) | `boosting.py`, `run_boosting.py` y soporte auxiliar de construcción; preprocesamiento para árboles | Rama paralela desde `main`; se conserva intacta. La entrega incorporó su implementación y una comparación posterior con 220,003 préstamos |
| [feat/exploracion-variables](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/feat/exploracion-variables) | Exploración completa y selección; `1445c66` | Memoria/tipos, cobertura temporal, faltantes, selección L1/permutación/RFE/PCA y RFECV; árbol con variables reducidas | `03_auditoria_tipos_memoria.ipynb`, `04_exploracion_final.ipynb`, `05_seleccion_variables.ipynb`, `06_arbol_30_variables.ipynb`, `Reduccion_dimensionalidad.ipynb` (8 notebooks acumulados) | Añade `avance.py`, `modelado.py`, workers de comparación de carga/tipos y reutiliza los modelos iniciales | Amplía `main`; su estado se conserva dentro del historial de `feat/modelado-final` |
| [feat/modelado-final](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/feat/modelado-final) | Comparación temporal y optimización; `b3b0754` | Logística, SVM y boosting, 26/59 variables, selección temporal, imputación, umbrales y sensibilidad; optimización definitiva | `07_entrenamiento.ipynb`, `08_busqueda_hiperparametros.ipynb`, experimentos `09`–`09g` (17 notebooks acumulados) | Reutiliza `modelado.py` y módulos de preparación; las búsquedas y comparaciones adicionales se implementaron en los notebooks de esta etapa | Continúa sobre exploración de variables y proporciona las configuraciones que recibe calibración |
| [feat/calibracion-evaluacion](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/feat/calibracion-evaluacion) | Calibración y evaluación histórica; `d166af7` | Sigmoide/isotónica, métricas probabilísticas, test 2017–2018 y errores por segmentos | `10_calibracion_probabilidades.ipynb`, `11_evaluacion_final.ipynb`, `12_analisis_errores.ipynb` (20 notebooks acumulados) | Añade `etapas_finales.py` y `analisis_errores.py` para calibradores, evaluación y segmentos | Continúa sobre modelado; precede a la consolidación de diez notebooks y al ejecutor final |
| [docs/entrega-final](https://github.com/Josu31506/Proyecto-ML-Grupo8/tree/docs/entrega-final) | Entrega integrada y documentación vigente | Diez notebooks consolidados, comparación inicial de muestra verificada y pipeline final entrenar/evaluar/predecir | Los diez archivos oficiales de la tabla principal | `run_final.py`, artefacto completo en `etapas_finales.py`, módulos reutilizables de comparaciones, rutas, reproducción y pruebas sintéticas | Contiene los commits originales de las cuatro ramas `feat/`; es la versión propuesta para revisión e integración futura con `main` |

`feat/exploracion-inicial` se creó como referencia al commit existente `b4901e3`: no contiene un commit nuevo ni una reconstrucción de archivos. Las otras tres ramas `feat/` se publicaron en sus commits originales. Publicar estas referencias no modifica los experimentos ni supone reejecutar sus resultados.

Las métricas deben interpretarse con el tamaño de muestra y los cortes de cada etapa. Los outputs preliminares de `gradient-boosting` se conservan como estaban, incluidas sus limitaciones de comparabilidad; la ejecución homogénea posterior de 220,003 está documentada en la entrega final. La reproducción del test del pipeline se confirma por la evaluación local documentada arriba, no por el hecho de publicar las ramas.

## Limitaciones conocidas

- Solo hay préstamos aceptados y con desenlace final observado. Excluir activos puede favorecer desenlaces tempranos en los años recientes; no se conoce la fecha individual de resolución.
- Los cortes ordenan por otorgamiento, pero no prueban que todas las etiquetas hubieran estado disponibles al entrenar en cada fecha simulada.
- El test 2017–2018 se consultó en investigaciones anteriores. La búsqueda y calibración finales usan desarrollo, pero no se puede describir todo el proyecto como si el test nunca hubiera sido observado.
- Las 59 variables fijas son exploratorias; el flujo final vuelve a seleccionar con los datos históricos de cada base. El 50 % es una regla conservadora elegida, no un límite universal ni el umbral con mayor promedio en todas las comparaciones.
- Las mejoras pequeñas y la elección entre varias configuraciones no demuestran significación estadística ni un óptimo universal. Las probabilidades y diferencias por segmento no prueban causalidad.
- Los artefactos locales están entrenados y sus métricas de test fueron reproducidas mediante evaluación explícita. Las pruebas sintéticas y esa coincidencia numérica no sustituyen una validación prospectiva independiente ni eliminan los sesgos de la población estudiada.
