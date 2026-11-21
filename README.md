# Predicción de incumplimiento de préstamos — Lending Club

Proyecto final del curso **Machine Learning**, **UTEC**, **Grupo 8**. Estimamos el riesgo individual de que un préstamo personal aceptado termine declarado incobrable. El resultado sirve de apoyo al análisis crediticio.

**Integrantes:** Gavino Navarrete Adrian Piero; Hernández Yataco, Josué Renzo; Rojas Cruz, José Alvaro; Sanchez Dominguez, Matias Sebastian.

**Pregunta predictiva del proyecto:** Dado un préstamo que Lending Club acaba de aprobar, y usando únicamente la información disponible el día del desembolso, ¿qué probabilidad hay de que termine declarado incobrable (`Charged Off`) en lugar de pagarse por completo (`Fully Paid`)?

Cada fila representa un préstamo. La variable objetivo es `Fully Paid = 0` y `Charged Off = 1`. El modelo principal es **HistGradientBoosting**, comparado con regresión logística y SVM lineal en el flujo final. El árbol de decisión forma parte de los experimentos iniciales.

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
├── data/
│   ├── raw/                        # Dataset local (excluido de Git)
│   │   └── accepted_2007_to_2018Q4.csv.gz
│   ├── README.md                   # Obtención del dataset y diccionario
│   └── Lending Club Data Dictionary Approved - ES.csv
├── notebooks/                      # Once notebooks oficiales
│   └── README.md                   # Orientación de lectura
├── src/                            # Preparación, experimentos y pipeline reutilizable
├── tests/                          # Ocho módulos de pruebas automatizadas
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

## Once notebooks oficiales

El orden sigue las decisiones del proyecto. Para usar el modelo final basta con el pipeline de más abajo; no es necesario ejecutar todos los experimentos.

| Nº | Archivo | Objetivo y experimentos | Decisión o resultado principal | Siguiente etapa |
|---:|---|---|---|---|
| 01 | [01_exploracion_inicial_baseline.ipynb](notebooks/01_exploracion_inicial_baseline.ipynb) | EDA, muestra, faltantes, outliers y regresiones logísticas iniciales | Muestra de 220,003; referencia sin indicadores y escenarios alternativos | Comparar relaciones no lineales |
| 02 | [02_arbol_decision.ipynb](notebooks/02_arbol_decision.ipynb) | Árbol libre/podado y grilla de 168 configuraciones | Comparación con logística usando los mismos periodos de la muestra | Comparar con un ensamble |
| 03 | [03_gradient_boosting.ipynb](notebooks/03_gradient_boosting.ipynb) | Consulta la grilla ejecutada de 108 configuraciones; reproducción explícita con muestra verificada y referencias guardadas | Ejecución de 220,003 guardada: ROC-AUC de validación de boosting 0.71556; incluye ambas variantes logísticas | Ampliar a todos los préstamos elegibles |
| 04 | [04_exploracion_completa.ipynb](notebooks/04_exploracion_completa.ipynb) | 97 columnas candidatas/auxiliares, tipos compactos, cobertura, correlaciones e imputación descriptiva | Primera selección fija de 26 variables con menos de 10 % de faltantes hasta 2012; anexo de carga/memoria | Entrenar con validación temporal |
| 05 | [05_svm_muestra.ipynb](notebooks/05_svm_muestra.ipynb) | SVM, logística, árbol y boosting con una matriz común sobre la muestra inicial | Primera ejecución de 220,003 guardada: SVM ROC-AUC 0.712178 en validación y 0.690612 en prueba; usa márgenes sin calibrar | Comparación complementaria; no sustituye el SVM final calibrado |
| 06 | [06_entrenamiento_busqueda_inicial.ipynb](notebooks/06_entrenamiento_busqueda_inicial.ipynb) | Modelos iniciales y primera búsqueda aleatoria con 26 variables | Cuatro cortes expansivos; referencia para las comparaciones posteriores | Revisar selección e imputación |
| 07 | [07_comparacion_variables_imputacion.ipynb](notebooks/07_comparacion_variables_imputacion.ipynb) | 26 variables, 59 exploratorias, selección por corte; mediana, indicadores y NaN nativo | Los indicadores no muestran una ventaja importante y consistente; las 59 fijas usan cobertura hasta 2016 y son exploratorias | Revisar el umbral de faltantes |
| 08 | [08_umbrales_sensibilidad.ipynb](notebooks/08_umbrales_sensibilidad.ipynb) | Umbrales 10/30/50/75 %; disponibilidad y sensibilidad 50 frente a 75 | Se adopta 50 % por cautela temporal. El 75 % tuvo mayor promedio, pero su ventaja ROC de boosting se concentró en un año | Optimizar con la regla elegida |
| 09 | [09_optimizacion_final.ipynb](notebooks/09_optimizacion_final.ipynb) | Configuraciones aleatorias y selección de variables dentro de cada entrenamiento | Fija parámetros de logística, SVM e HistGradientBoosting; no pretende un óptimo universal | Evaluar probabilidades |
| 10 | [10_calibracion.ipynb](notebooks/10_calibracion.ipynb) | Original, sigmoide e isotónica con base/calibración/evaluación cronológicas | Se elige sigmoide para los tres modelos usando desarrollo | Evaluar las configuraciones fijadas |
| 11 | [11_evaluacion_final.ipynb](notebooks/11_evaluacion_final.ipynb) | Test histórico, intervalos, curvas y errores por segmentos | Comparación final calibrada e interpretación de sus límites | Inferencia mediante artefactos guardados |

La muestra inicial usa entrenamiento **2007–2014**, validación **2015–2016** y prueba **2017–2018**. El flujo posterior utiliza todos los préstamos elegibles y reglas ajustadas por periodo. Sus tamaños y puntuaciones no deben mezclarse como si pertenecieran al mismo experimento.

## Código reutilizable de `src/`

| Módulo | Entrada y responsabilidad | Salida y consumidores |
|---|---|---|
| [config.py](src/config.py) | Semilla, rutas, etiquetas, cortes y constantes de escenarios | Configuración compartida; los umbrales del experimento inicial y del flujo final son distintos |
| [data.py](src/data.py) | CSV original; auditoría, normalización, muestra proporcional por bloques y partición anual | Muestra y auditoría para baseline, árbol y boosting inicial |
| [experimento_inicial.py](src/experimento_inicial.py) | Auditoría histórica o CSV; carga, muestreo y particiones comunes para 02–03 y 05 | Modo consulta sin entrenamiento; modo reproducción con una llamada al muestreador y referencias verificadas |
| [features.py](src/features.py) | Tablas de préstamos; excluye auxiliares, identificadores, texto y variables posteriores; prepara fechas y cobertura | Predictoras y preprocesadores usados por los escenarios y etapas finales |
| [train.py](src/train.py) | Preprocesador y tipo de referencia | Constructores de modelos iniciales y tabla de coeficientes; auxiliar del baseline |
| [evaluate.py](src/evaluate.py) | Etiquetas y puntuaciones | Métricas, umbral elegido en validación y exportación JSON de experimentos iniciales |
| [experimentos.py](src/experimentos.py) | Particiones temporales y escenarios logísticos | Matrices, modelos y métricas comparables de baseline; usado por notebooks y ejecutores iniciales |
| [arbol.py](src/arbol.py) | Particiones y grilla del árbol | Preparación, búsqueda, modelo, métricas e importancias; usado por notebook 02 y el ejecutor histórico `run_arbol` |
| [visualizacion_arbol.py](src/visualizacion_arbol.py) | Resultados del experimento de árbol | Cuatro gráficos del notebook 02; no carga datos ni entrena modelos |
| [boosting.py](src/boosting.py) | Particiones y grilla de HistGradientBoosting inicial | Preparación, búsqueda, modelo e importancia por permutación; usado por notebook 03 y el ejecutor histórico `run_boosting` |
| [comparabilidad_muestra.py](src/comparabilidad_muestra.py) | CSV, muestra y referencias iniciales | Reconstrucción de IDs/orden y huellas de particiones; protege la comparación inicial |
| [modelado.py](src/modelado.py) | Dataset, periodos y modelos candidatos | Tipos auditados, cortes, imputación y métricas de desarrollo; usado por comparaciones y búsquedas |
| [svm_muestra.py](src/svm_muestra.py) | Muestra inicial y escenario sin indicadores | Cuatro modelos con matriz común, márgenes SVM, umbral F1 en validación y exportación aislada; notebook 05 |
| [etapas_finales.py](src/etapas_finales.py) | Dataset o matrices, parámetros congelados y periodos | Selección al 50 %, preparación, modelos, calibradores, métricas y `PipelineFinal`; usado por notebooks 10–11 y ejecutor final |
| [run_final.py](src/run_final.py) | Argumentos de entrenamiento, evaluación o predicción | Artefactos completos, evaluaciones independientes de ejecución o probabilidades de préstamos nuevos |
| [analisis_errores.py](src/analisis_errores.py) | Probabilidades, etiquetas y segmentos del CSV | Verificación de alineación y métricas globales/por segmento; usado por notebook 11 y evaluación del pipeline |
| [rutas.py](src/rutas.py) | Ubicación del proyecto o variable `PROYECTO_ML_RAIZ` | Resolución común de la raíz para notebooks y scripts |
| [reproduccion_notebooks.py](src/reproduccion_notebooks.py) | Modo de notebook, referencias CSV/JSON y destino | Consulta sin ajustes; manifiesto y comparación automática en reproducciones |
| [reproducir.py](src/reproducir.py) | Nombre de experimento y destino nuevo | Despacha comparaciones a carpetas nuevas bajo `outputs/reproducciones/` |
| [preparacion_temporal.py](src/preparacion_temporal.py) | Predictoras y fecha de otorgamiento | Transformación compartida de fechas y tipos para las comparaciones |
| [entrenamiento_inicial.py](src/entrenamiento_inicial.py) | Dataset y carpeta de salida | Referencias con 26 variables; regeneración desde notebook 06 |
| [comparacion_variables.py](src/comparacion_variables.py) | Dataset y reglas de selección | Métricas 26/59/selección temporal, con/sin indicadores; notebook 07 |
| [comparacion_imputacion.py](src/comparacion_imputacion.py) | Dataset y tratamientos de faltantes | Comparación mediana/indicadores/NaN nativo; notebook 07 |
| [comparacion_lineales.py](src/comparacion_lineales.py) | Dataset y configuraciones lineales ya elegidas | Comparaciones de logística/SVM por variables y tratamiento; notebook 07 |
| [comparacion_umbrales.py](src/comparacion_umbrales.py) | Dataset y cuatro umbrales fijos | Selección por entrenamiento y métricas por modelo/año; notebook 08 |
| [sensibilidad_faltantes.py](src/sensibilidad_faltantes.py) | Dataset y tablas de comparación de umbrales | Cobertura, diferencias anuales e intervalos 50/75; notebook 08 |
| [auditoria_memoria.py](src/auditoria_memoria.py) | CSV completo y destino | Comprobación de tipos sin pérdida y comparación de carga; notebook 04 |
| [benchmark_worker.py](src/benchmark_worker.py) | Configuración de lectura, método y destino | Medición de carga directa/bloques y huella de igualdad; auxiliar de auditoría |
| [run_baseline.py](src/run_baseline.py) | CSV y objetivo de muestra | Ejecuta el baseline inicial y genera métricas/figuras |
| [run_arbol.py](src/run_arbol.py) | CSV y objetivo de muestra | Interfaz histórica de `main`; los notebooks actuales no dependen de ella |
| [run_boosting.py](src/run_boosting.py) | CSV y objetivo de muestra | Interfaz histórica de `main`; los notebooks actuales no dependen de ella |

Los ejecutores históricos `src/run_arbol.py` y `src/run_boosting.py` se conservan sin cambios desde `main` para evitar perder interfaces previas. Pueden escribir en los destinos históricos del proyecto: para reproducir los experimentos en carpetas nuevas, utiliza los controles de los notebooks 02 y 03. El pipeline definitivo sigue siendo `src/run_final.py`. El guion `reports/presentacion_esquema.md` también se conserva como documento de apoyo histórico; el informe final es la referencia académica revisada.

## Metodología

```text
Dataset original → filtrado y limpieza → objetivo binario → control de leakage → separación temporal → selección de variables → preprocesamiento → entrenamiento → calibración → evaluación → predicción
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

Las pruebas ejecutadas con `python -m unittest discover -s tests -v` finalizaron con éxito.

Son pruebas con datos sintéticos de particiones, prevención de leakage, preprocesamiento, predicción, calibración, serialización y evaluación explícita. La salida de consola acredita la ejecución local reportada; no se presenta como un registro de GitHub Actions ni sustituye la evaluación real de Lending Club.

Para repetir la evaluación histórica de los modelos guardados con la ubicación de datos recomendada:

```powershell
python -m src.run_final --modo evaluar --modelos models/final_01 --datos data/raw/accepted_2007_to_2018Q4.csv.gz --conjunto test --confirmar-test
```

| Modelo calibrado con sigmoide | ROC-AUC | PR-AUC (AP) | Brier Score | Log Loss | n |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.692591 | 0.360386 | 0.155911 | 0.485014 | 225611 |
| SVM lineal | 0.691777 | 0.359876 | 0.155972 | 0.486804 | 225611 |
| HistGradientBoosting | 0.703213 | 0.377905 | 0.153642 | 0.474278 | 225611 |

El test corresponde a 2017–2018 y su prevalencia de incumplimiento es 0.212822. La evidencia persistente local se encuentra en `outputs/pipeline_final/test_20261117_195412_266845/metricas.csv` y `alcance.json`, excluidos de Git junto con sus predicciones individuales. Se compararon las cuatro métricas, la prevalencia y n de los tres modelos con [la referencia histórica](outputs/evaluacion_final/metricas_finales_test.csv): los 18 campos numéricos coinciden en los CSV y, por tanto, a los seis decimales presentados. No se recalcularon predicciones ni se repitió el test durante esta revisión documental.

Esto confirma la reproducción numérica de las métricas mediante los artefactos serializados, no una validación externa ni la identidad fila a fila de todas las predicciones. El test ya había sido consultado durante el desarrollo histórico; siguen vigentes la selección de préstamos con desenlace observado, las limitaciones de maduración de etiquetas y la ausencia de evaluación prospectiva estrictamente independiente.

## Instalación y datos

Ejecutar desde la raíz del repositorio, por ejemplo en PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Descargar el dataset siguiendo las instrucciones de [data/README.md](data/README.md). Extraer del ZIP de Kaggle el archivo `accepted_2007_to_2018Q4.csv.gz` (o su versión `.csv`) y colocarlo preferentemente en `data/raw/`. El código también reconoce archivos en `data/`, en la raíz del proyecto y en Downloads, o permite especificar una ruta con `--datos`. El CSV descomprimido ocupa aproximadamente 1.6 GB; algunas etapas requieren memoria adicional. Los datos originales y los artefactos entrenados se excluyen de Git.


## Pipeline final reproducible

El proyecto implementa un pipeline de Machine Learning mediante el módulo `src.run_final`, que permite **entrenar, evaluar y utilizar los modelos definitivos** sin necesidad de ejecutar nuevamente los once notebooks de investigación.

Los notebooks documentan el desarrollo experimental: exploración, comparación de algoritmos, selección de características, búsqueda de hiperparámetros, calibración y evaluación histórica. En cambio, `src.run_final` reutiliza las configuraciones seleccionadas para ejecutar el procedimiento definitivo.

El pipeline tiene tres modos de operación:

| Modo | Función principal | Resultado |
| --- | --- | --- |
| `entrenar` | Ajusta los modelos definitivos y sus calibradores a partir del dataset original. | Modelos serializados `.joblib` y `manifest.json`. |
| `evaluar` | Calcula métricas sobre datos históricos utilizando modelos previamente entrenados. | Métricas, predicciones, gráficos y análisis de errores. |
| `predecir` | Estima el riesgo de incumplimiento de nuevos préstamos. | CSV con probabilidades individuales de `Charged Off`. |

Estos modos se ejecutan por separado. El entrenamiento debe realizarse antes de utilizar los otros dos modos cuando todavía no se dispone de los artefactos del modelo.

### 1. Entrenar y calibrar los modelos finales

**Objetivo:** construir los tres modelos definitivos utilizando el dataset original y guardar todos los componentes necesarios para reutilizarlos posteriormente.

El entrenamiento emplea las configuraciones seleccionadas durante la investigación; **no realiza una nueva búsqueda de hiperparámetros ni vuelve a comparar configuraciones alternativas**.


#### Datos de entrada

Para entrenar los modelos es necesario descargar el dataset original de Lending Club desde [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

Una vez descargado, extraer del ZIP el archivo `accepted_2007_to_2018Q4.csv.gz` y colocarlo en `data/raw/`. También se admite `accepted_2007_to_2018Q4.csv` sin comprimir.

El dataset no se incluye en GitHub. El programa no procesa directamente archivos `.zip`, pero puede leer el `.csv.gz` sin descomprimirlo. Para más detalles, consultar [data/README.md](data/README.md).

#### Ejecución

Desde la raíz del repositorio:

```powershell
python -m src.run_final --modo entrenar --datos data/raw/accepted_2007_to_2018Q4.csv.gz --salida models/final_01
```

El pipeline carga el archivo desde la ruta indicada, filtra los préstamos elegibles y utiliza las particiones temporales establecidas para entrenar y calibrar los modelos definitivos.

Aunque el cargador admite otras ubicaciones y archivos `.csv.gz`, se recomienda mantener los datos originales en `data/raw/` para conservar una estructura organizada y reproducible.

#### ¿Qué hace internamente?

El procedimiento sigue estas etapas:

1. **Verifica la configuración definitiva.** Comprueba que los hiperparámetros, la semilla, el criterio de selección de características y el método de calibración coincidan con las configuraciones históricas guardadas.

2. **Carga y filtra los datos.** Conserva los préstamos elegibles con los estados `Fully Paid` y `Charged Off`, utilizando el conjunto de desarrollo hasta 2016.

3. **Realiza la separación temporal.** Utiliza los préstamos de 2007–2015 para ajustar los modelos y reserva los préstamos de 2016 para calibrar sus probabilidades.

4. **Selecciona y prepara las características.** Aplica el criterio definitivo de hasta 50 % de valores faltantes, ajustando la selección, imputación, codificación de categorías y escalado únicamente con los datos de entrenamiento base.

5. **Entrena tres modelos.** Ajusta regresión logística, SVM lineal e HistGradientBoosting con sus hiperparámetros definitivos.

6. **Calibra las probabilidades.** Utiliza los datos de 2016 para ajustar un calibrador sigmoide por modelo, sin volver a entrenar el estimador base con ese periodo.

7. **Guarda los artefactos.** Serializa los modelos completos y genera un manifiesto con la información necesaria para su carga y verificación.

#### Archivos generados

```text
models/
└── final_01/
    ├── regresion_logistica.joblib
    ├── svm_lineal.joblib
    ├── gradient_boosting.joblib
    └── manifest.json
```

Cada archivo `.joblib` conserva el modelo correspondiente junto con la información de selección de variables, preprocesamiento y calibración.

El archivo `manifest.json` identifica el modelo principal, las rutas y huellas SHA-256 de los artefactos, sus configuraciones, versiones de dependencias y metadatos de entrenamiento.

**El modo `entrenar` no evalúa automáticamente el conjunto de prueba de 2017–2018.** Su función es generar los modelos definitivos, no medir nuevamente su rendimiento histórico.

#### Consideraciones

Los artefactos `.joblib` no se publican en GitHub. Por tanto, quien descargue el repositorio por primera vez deberá entrenar los modelos si desea utilizar los modos `evaluar` o `predecir`.

La carpeta de salida debe ser nueva. Si `models/final_01/` ya existe, el programa rechazará sobrescribirla. Puede utilizarse otra ruta, como `models/final_02/`, o no indicar `--salida` para generar una carpeta con nombre fechado.

Los artefactos serializados requieren versiones compatibles de las dependencias registradas durante su entrenamiento.

### 2. Evaluar los modelos entrenados

**Objetivo:** medir el rendimiento de los modelos definitivos sobre préstamos históricos con desenlace conocido, sin volver a entrenarlos ni modificar sus calibradores.

Este modo permite comprobar la capacidad discriminativa, la calidad de las probabilidades y el comportamiento de los modelos al utilizar un umbral de clasificación.

#### Datos de entrada

Se necesitan:

- El dataset original `data/raw/accepted_2007_to_2018Q4.csv.gz` (o su versión CSV descomprimida).
- La carpeta que contiene los tres modelos `.joblib` y el archivo `manifest.json`.
- La indicación del periodo que se desea evaluar.

El programa comprueba la integridad de los artefactos mediante sus huellas SHA-256 y verifica la compatibilidad de las versiones de sus dependencias.

#### Evaluación descriptiva del periodo de calibración

```powershell
python -m src.run_final --modo evaluar --modelos models/final_01 --datos data/raw/accepted_2007_to_2018Q4.csv.gz --conjunto validacion
```

Esta instrucción calcula las métricas sobre los préstamos de **2016**.

Es importante interpretar correctamente el resultado: el año 2016 ya se utilizó para ajustar los calibradores sigmoidales. Por ello, esta evaluación es descriptiva y **no constituye una evaluación independiente** de las probabilidades calibradas.

#### Evaluación del conjunto de prueba histórico

```powershell
python -m src.run_final --modo evaluar --modelos models/final_01 --datos data/raw/accepted_2007_to_2018Q4.csv.gz --conjunto test --confirmar-test
```

Esta instrucción utiliza los préstamos elegibles originados entre **2017 y 2018**, separados del ajuste base y de la calibración definitiva.

El argumento `--confirmar-test` es obligatorio para evitar consultar accidentalmente el conjunto de prueba.

#### ¿Qué hace internamente?

1. **Carga los modelos guardados.** Recupera sus preprocesadores, estimadores y calibradores sin reajustarlos.

2. **Selecciona el periodo solicitado.** Filtra los préstamos elegibles de 2016 o de 2017–2018, según el argumento `--conjunto`.

3. **Aplica las transformaciones aprendidas.** Utiliza las características y parámetros de preparación conservados durante el entrenamiento.

4. **Calcula probabilidades calibradas.** Obtiene una probabilidad estimada de `Charged Off` para cada préstamo y para cada uno de los tres modelos.

5. **Calcula métricas predictivas.** Incluye ROC-AUC, Average Precision, Brier Score y Log Loss.

6. **Analiza la clasificación.** Aplica el umbral descriptivo de 0.5 para estudiar el comportamiento de los modelos y sus errores.

7. **Genera resultados y gráficos.** Guarda las métricas, las predicciones, el análisis de errores por segmentos y las curvas de evaluación.

#### Archivos generados

Cada evaluación crea una carpeta nueva dentro de `outputs/pipeline_final/`, con archivos como:

```text
outputs/
└── pipeline_final/
    └── test_YYYYMMDD_HHMMSS/
        ├── auditoria_estados.csv
        ├── metricas.csv
        ├── predicciones.csv.gz
        ├── clasificacion_0_5.csv
        ├── errores_segmentos.csv
        ├── evaluacion.png
        └── alcance.json
```

El nombre de la carpeta es ilustrativo; el programa utiliza un identificador temporal. `auditoria_estados.csv` se genera cuando se evalúa el test.

Los archivos cumplen funciones distintas:

| Archivo | Contenido |
| --- | --- |
| `metricas.csv` | Métricas de rendimiento de los tres modelos. |
| `predicciones.csv.gz` | Probabilidades estimadas y etiquetas reales de las observaciones evaluadas. |
| `clasificacion_0_5.csv` | Resultados de clasificación utilizando el umbral 0.5. |
| `errores_segmentos.csv` | Métricas y errores desagregados por segmentos disponibles. |
| `evaluacion.png` | Curvas ROC, Precision–Recall y calibración. |
| `alcance.json` | Periodo evaluado, interpretación, umbral y versiones de ejecución. |
| `auditoria_estados.csv` | Conteos de estados inspeccionados durante la evaluación del test. |

**El umbral de 0.5 es exclusivamente descriptivo** y no representa una política de aprobación de créditos optimizada.

La evaluación histórica tampoco debe utilizarse para modificar hiperparámetros, seleccionar variables o elegir nuevamente la calibración a partir de los resultados del test.

### 3. Predecir el riesgo de nuevos préstamos

**Objetivo:** utilizar el modelo principal previamente entrenado para estimar la probabilidad de que nuevos préstamos terminen declarados incobrables.

Este modo corresponde a la utilización del modelo para inferencia. A diferencia de la evaluación, **no necesita conocer el desenlace real de los préstamos ni disponer del dataset histórico original**.

#### Requisitos previos

Antes de generar predicciones deben existir los artefactos del entrenamiento definitivo.

Si los modelos todavía no se han generado, primero debe ejecutarse el modo `entrenar` con el dataset original de Lending Club.

Una vez guardados, los modelos pueden reutilizarse para realizar múltiples predicciones sin repetir el entrenamiento.

#### Datos de entrada

Se necesita un archivo CSV con las características de los nuevos préstamos.

Por ejemplo:

`nuevos_prestamos.csv`

El archivo debe contener todas las columnas indicadas en `columnas_entrada` del manifiesto del modelo principal, conservando sus nombres originales.

Las columnas son obligatorias en el esquema, aunque sus valores individuales pueden estar ausentes. No es necesario incluir la etiqueta `loan_status`, la variable objetivo `default` ni información de pagos posteriores al otorgamiento.

Para las fechas de entrada se utiliza el formato esperado por el proyecto, por ejemplo `Jan-2016`. Las columnas adicionales del CSV se ignoran.

#### Ejecución

```powershell
python -m src.run_final --modo predecir --modelos models/final_01 --datos nuevos_prestamos.csv
```

#### ¿Qué hace internamente?

1. **Carga el modelo principal.** Según el manifiesto definitivo, utiliza HistGradientBoosting.

2. **Verifica las columnas de entrada.** Comprueba que el CSV incluya todas las características requeridas.

3. **Procesa los datos por bloques.** Lee hasta 10,000 filas por bloque para evitar cargar innecesariamente todo el archivo de predicción.

4. **Aplica el preprocesamiento guardado.** Reutiliza las variables seleccionadas, medianas, categorías, escaladores y transformaciones aprendidas durante el entrenamiento.

5. **Calcula probabilidades calibradas.** Obtiene para cada préstamo una estimación entre 0 y 1 de pertenecer a la clase `Charged Off`.

6. **Guarda las predicciones.** Genera un archivo CSV nuevo con la posición de cada registro y su probabilidad estimada.

#### Archivo generado

Por defecto, el programa crea un CSV con nombre fechado dentro de `outputs/predicciones/`.

El archivo contiene dos columnas:

| Columna | Significado |
| --- | --- |
| `fila_entrada` | Posición de la observación en el archivo de entrada, comenzando desde cero. |
| `probabilidad_charged_off` | Probabilidad estimada de que el préstamo termine declarado incobrable. |

Por ejemplo, una probabilidad de `0.30` representa una estimación de riesgo del 30 % bajo las condiciones y limitaciones del modelo.

También puede indicarse una ruta de salida:

```powershell
python -m src.run_final --modo predecir --modelos models/final_01 --datos nuevos_prestamos.csv --salida outputs/predicciones_nuevas_01.csv
```

El archivo de salida debe ser nuevo; el programa no sobrescribe predicciones existentes.

**Este modo no entrena, no recalibra y no evalúa métricas predictivas**, porque los préstamos nuevos no necesitan disponer de una etiqueta real.

Las probabilidades generadas no constituyen decisiones automáticas de aprobación o rechazo de crédito. Su uso está limitado por las características de la población de entrenamiento, la calibración y las advertencias metodológicas del informe.

### 4. Resumen del flujo de utilización

Para una persona que descarga el proyecto por primera vez, el procedimiento completo es:

```text
Dataset original de Lending Club
              |
              v
       Modo ENTRENAR
              |
              v
  Modelos .joblib + manifest.json
              |
       +------+------+
       |             |
       v             v
  Modo EVALUAR   Modo PREDECIR
       |             |
       v             v
Métricas y       Probabilidades
análisis de      individuales de
errores          nuevos préstamos
```

La secuencia depende del objetivo:

- **Revisar los resultados de la investigación:** consultar el informe, los notebooks y los archivos históricos de `outputs/`. No es necesario entrenar nuevamente.
- **Reproducir el modelo definitivo:** descargar el dataset y ejecutar `entrenar`.
- **Recalcular sus métricas históricas:** disponer de los modelos guardados y ejecutar `evaluar`.
- **Estimar riesgos de préstamos nuevos:** disponer de los modelos guardados y ejecutar `predecir` con un CSV compatible.

Los modos `evaluar` y `predecir` reutilizan los mismos artefactos, pero tienen objetivos distintos: el primero mide el rendimiento con etiquetas conocidas y el segundo genera probabilidades para observaciones cuya etiqueta no es necesaria.

### 5. Parámetros de la interfaz

| Argumento | Uso |
| --- | --- |
| `--modo` | Selecciona `entrenar`, `evaluar` o `predecir`. |
| `--datos` | Ruta del CSV. Opcional en entrenamiento si el archivo puede localizarse automáticamente; obligatorio en evaluación y predicción. |
| `--modelos` | Carpeta con `manifest.json` y archivos `.joblib`. Obligatorio en evaluación y predicción. |
| `--salida` | Ruta nueva para guardar modelos, resultados de evaluación o un CSV de predicciones. |
| `--conjunto` | Selecciona `validacion` (2016) o `test` (2017–2018) en el modo de evaluación. |
| `--confirmar-test` | Autoriza explícitamente la evaluación del conjunto de prueba histórico. |

Para consultar todos los argumentos:

```powershell
python -m src.run_final --help
```

El pipeline protege los destinos existentes para evitar sobrescribir artefactos y resultados previos.


## Verificación de la búsqueda de hiperparámetros

La optimización de hiperparámetros puede revisarse y reproducirse mediante los notebooks 06 y 09.

El notebook 06 documenta la búsqueda aleatoria inicial utilizando 26 características, mientras que el notebook 09 implementa la optimización definitiva con selección temporal de características y un umbral máximo del 50 % de faltantes.

La búsqueda definitiva utiliza cuatro cortes de validación temporal (2013–2016), una semilla fija y las tres familias de modelos: regresión logística, SVM lineal e HistGradientBoosting.

### Consultar los resultados históricos

Abrir `notebooks/09_optimizacion_final.ipynb` y mantener:

```python
REPRODUCIR = False
```

Este modo permite examinar los hiperparámetros seleccionados, las métricas por configuración y año, y las comparaciones de rendimiento sin volver a entrenar modelos.

Los resultados de referencia se encuentran en `outputs/optimizacion_final/`.

### Reproducir la búsqueda completa

Para ejecutar nuevamente el procedimiento:

```python
REPRODUCIR = True
RUTA_DATOS = "data/raw/accepted_2007_to_2018Q4.csv.gz"
```

Reiniciar el kernel y ejecutar todas las celdas en orden.

El notebook vuelve a evaluar las configuraciones definidas mediante los mismos cortes temporales y guarda los resultados en una carpeta nueva dentro de `outputs/reproducciones/`.

Al finalizar, compara los archivos generados con las referencias históricas y presenta los estados de verificación correspondientes.

La comparación utiliza tolerancias numéricas y comprueba también la estructura de las tablas y los parámetros disponibles. Las diferencias se reportan explícitamente; la coincidencia de métricas agregadas no garantiza, por sí sola, la identidad de todas las observaciones utilizadas.

Esta reproducción no modifica los hiperparámetros del pipeline definitivo ni consulta el conjunto de prueba 2017–2018.

## Pruebas automatizadas

El proyecto también incorpora pruebas automatizadas para verificar el funcionamiento del software y los controles de reproducibilidad.

Estas pruebas son independientes de la evaluación predictiva histórica. Su objetivo es detectar errores de implementación, no volver a medir el rendimiento de los modelos sobre Lending Club.

### Ejecución

Desde la raíz del repositorio:

```powershell
python -m unittest discover -s tests -v
```

### ¿Qué verifican las pruebas?

La suite incluye comprobaciones sobre:

- **Preparación y filtrado de datos:** particiones temporales, selección de estados elegibles y exclusión de información posterior al otorgamiento.
- **Preprocesamiento:** selección de características e imputación ajustadas exclusivamente con los datos de entrenamiento, además del tratamiento de categorías nuevas.
- **Modelos y predicciones:** funcionamiento con datos sintéticos, probabilidades dentro del intervalo esperado e inferencia sin disponer de la variable objetivo.
- **Almacenamiento:** serialización, carga de artefactos, manifiestos y compatibilidad de configuraciones.
- **Seguridad de ejecución:** protección contra sobrescritura de resultados y exigencia de confirmación antes de evaluar el test.
- **Reproducibilidad de notebooks:** controles de consulta y reproducción, separación de salidas y comparación de resultados CSV/JSON.
- **Experimentos iniciales:** comprobaciones de carga compartida, árbol de decisión, Gradient Boosting y SVM sobre muestras.

En la versión actual de `main`, los ocho archivos de `tests/` definen **54 métodos de prueba** (`test_*`), según una revisión estática del código. Para verificar cuántas pruebas descubre y supera efectivamente la suite en un entorno local, debe ejecutarse el comando anterior. Las ejecuciones exitosas documentadas durante el cierre académico no constituyen una nueva ejecución de la versión actual.

Estas pruebas utilizan principalmente datos sintéticos y referencias guardadas. No ejecutan el entrenamiento completo de los modelos sobre Lending Club ni constituyen una nueva evaluación independiente sobre el conjunto de prueba.

## Resultados, informe y publicación

[outputs/README.md](outputs/README.md) describe métricas y configuraciones por etapa. El [informe final](reports/informe_final.md) desarrolla resultados y límites; el [guion de exposición](reports/presentacion_esquema.md) organiza la presentación. Las figuras están en [reports/figures/](reports/figures/) y en las carpetas de resultados correspondientes.
