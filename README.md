# Lending Club — predicción de `Charged Off`

Proyecto final de Machine Learning del Grupo 8 (UTEC). Estima el riesgo de que un préstamo aceptado termine como `Charged Off`, usando variables asociadas al momento de originación. El resultado se plantea como apoyo analítico, no como decisión automática sobre una persona.

**Integrantes:** Gavino Navarrete Adrian Piero; Hernández Yataco, Josué Renzo; Rojas Cruz, José Alvaro; Sanchez Dominguez, Matias Sebastian.

## Resultados finales disponibles

Los resultados siguientes pertenecen a los modelos calibrados en validación temporal y evaluados en el test 2017–2018. El test incluyó 225,611 préstamos con estado final binario (`Fully Paid` o `Charged Off`), con 21.28 % `Charged Off`.

| Modelo | ROC-AUC | AP (PR-AUC) | Brier | Log Loss |
|---|---:|---:|---:|---:|
| Regresión logística | 0.6926 | 0.3604 | 0.1559 | 0.4850 |
| SVM lineal | 0.6918 | 0.3599 | 0.1560 | 0.4868 |
| HistGradientBoosting | **0.7032** | **0.3779** | **0.1536** | **0.4743** |

HistGradientBoosting fue el mejor de estos tres modelos en las métricas reportadas; las diferencias son moderadas y dependen de la población con desenlace observado. El archivo contiene 938,821 préstamos emitidos en 2017–2018, pero solo 225,611 tenían uno de los dos estados finales seleccionados. Los préstamos activos y atrasados se excluyeron. No hay fecha individual de resolución; por tanto, el test no representa automáticamente todos los préstamos emitidos en esos años.

**Limitación de trazabilidad del test:** experimentos históricos del baseline, el árbol y reducción de dimensionalidad ya consultaron métricas en 2017–2018. La calibración final y la búsqueda final se seleccionaron con validación, pero no es correcto afirmar que el test nunca fue observado durante todo el proyecto.

Resultados reproducibles guardados: [`outputs/evaluacion_final/metricas_finales_test.csv`](outputs/evaluacion_final/metricas_finales_test.csv), [`outputs/evaluacion_final/intervalos_bootstrap_95.csv`](outputs/evaluacion_final/intervalos_bootstrap_95.csv), [`outputs/calibracion_probabilidades/metricas_promedio.csv`](outputs/calibracion_probabilidades/metricas_promedio.csv) y [`outputs/optimizacion_final/configuracion_reproducible.json`](outputs/optimizacion_final/configuracion_reproducible.json).

## Dataset y objetivo

- **Fuente:** Lending Club Loan Data, 2007–2018. Instrucciones para obtener el CSV están en [`data/README.md`](data/README.md).
- **Archivo:** `accepted_2007_to_2018Q4.csv` o `.csv.gz`, no versionado en Git.
- **Universo binario:** `Fully Paid = 0`, `Charged Off = 1`; la primera auditoría contó 1,345,310 casos de ambos estados. Otros estados no son desenlaces finales para esta tarea.
- **Variables posteriores al otorgamiento:** pagos, recuperaciones, saldos actualizados, consultas/score actualizado, dificultades y acuerdos se excluyen por posible fuga de información.
- El escenario final no incluye el bloque de calificación/precio de Lending Club; confirmar disponibilidad temporal de cada campo antes de cualquier uso futuro.

## Recorrido recomendado

| Orden | Notebook | Propósito |
|---:|---|---|
| 1 | [`01_exploracion_inicial.ipynb`](notebooks/01_exploracion_inicial.ipynb) | EDA y baseline de la primera entrega, con muestra |
| 2 | [`03_auditoria_tipos_memoria.ipynb`](notebooks/03_auditoria_tipos_memoria.ipynb) | Auditoría de tipos y comparación de carga |
| 3 | [`04_exploracion_final.ipynb`](notebooks/04_exploracion_final.ipynb) | Cobertura y análisis del archivo completo |
| 4 | [`02_avance_modelos.ipynb`](notebooks/02_avance_modelos.ipynb), [`03_arbol_decision.ipynb`](notebooks/03_arbol_decision.ipynb), [`04_gradient_boosting.ipynb`](notebooks/04_gradient_boosting.ipynb) | Avance y modelos exploratorios/históricos |
| 5 | [`05_seleccion_variables.ipynb`](notebooks/05_seleccion_variables.ipynb), [`Reduccion_dimensionalidad.ipynb`](notebooks/Reduccion_dimensionalidad.ipynb), [`06_arbol_30_variables.ipynb`](notebooks/06_arbol_30_variables.ipynb) | Experimentos de reducción; algunos artefactos históricos examinaron test |
| 6 | [`07_entrenamiento.ipynb`](notebooks/07_entrenamiento.ipynb), [`08_busqueda_hiperparametros.ipynb`](notebooks/08_busqueda_hiperparametros.ipynb), [`09g_optimizacion_final.ipynb`](notebooks/09g_optimizacion_final.ipynb) | Validación temporal y búsqueda de parámetros |
| 7 | [`10_calibracion_probabilidades.ipynb`](notebooks/10_calibracion_probabilidades.ipynb), [`11_evaluacion_final.ipynb`](notebooks/11_evaluacion_final.ipynb) | Calibración y evaluación guardada final |
| 8 | [`12_analisis_errores.ipynb`](notebooks/12_analisis_errores.ipynb) | Errores por año, plazo, ingreso y tipo de solicitud |

Los notebooks `09b` a `09f` documentan comparaciones de variables, imputación, indicadores, umbrales de faltantes y análisis de sensibilidad. Se conservan para la trazabilidad experimental; no todos son etapas del pipeline final.

## Evolución de modelos

| Modelo o estrategia | Papel | Comparabilidad |
|---|---|---|
| Clase mayoritaria | Referencia simple | No detecta `Charged Off`; ROC-AUC 0.5 |
| Regresión logística | Baseline estadístico y modelo final lineal | El baseline inicial usa muestra/preparación diferentes |
| Árbol de decisión | Modelo exploratorio no lineal | Resultados de la primera etapa |
| Gradient Boosting inicial | Experimento histórico de la rama `gradient-boosting` | Muestra ≈5,002 y un corte temporal; no comparable directamente con el final |
| SVM lineal | Modelo final basado en margen | Mismas etapas finales de calibración/evaluación |
| HistGradientBoosting | Ensamble optimizado seleccionado | Mejor resultado entre los tres modelos finales evaluados |

La rama `gradient-boosting` conserva su notebook y código. Su JSON de resultados también contiene una tabla de árbol con tamaños de muestra distintos; úsese solo como referencia histórica hasta regenerar una comparación coherente. El análisis está documentado en [`reports/auditoria_integracion.md`](reports/auditoria_integracion.md).

## Cortes y tratamiento de faltantes

La búsqueda final usa validación expansiva: entrenar hasta 2012→validar 2013; hasta 2013→2014; hasta 2014→2015; hasta 2015→2016. La selección de predictores se realiza en cada conjunto base con límite de 50 % de faltantes: 26 originales para el primer corte y 60 para los otros tres; ajuste final de 60. Las medianas y categorías se aprenden solo con la base. El modelo final no usa indicadores de faltantes.

El límite histórico de 95 %, el criterio exploratorio menor de 10 % y el 50 % del flujo final son decisiones de etapas diferentes. Los números de una etapa no deben atribuirse a otra.

## Reproducción

### Preparar el entorno

```bash
git clone https://github.com/Josu31506/Proyecto-ML-Grupo8.git
cd Proyecto-ML-Grupo8
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`. Luego:

```bash
pip install -r requirements.txt
```

Coloca el archivo descargado de acuerdo con [`data/README.md`](data/README.md), idealmente en `data/raw/`. El dataset no está versionado.

### Ejecutar código

```bash
python -m src.run_baseline
python -m unittest discover -s tests -v
```

El primer comando reproduce el baseline de muestra, **no** la evaluación calibrada final. Para evaluación final, sigue los notebooks 09g–11; requieren el CSV completo y una ejecución con memoria y tiempo suficientes. El análisis de errores puede ejecutarse con `python -m src.analisis_errores` después de generar `outputs/evaluacion_final/predicciones_test.csv.gz`.

Durante esta integración se verificaron imports y pruebas en Python 3.14.0 con pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.8.0 y matplotlib 3.10.8. No se verificó una instalación limpia desde `requirements.txt`; las versiones anotadas en notebooks pueden diferir.

## Estructura principal

```text
notebooks/       exploración, modelos, validación, calibración y errores
src/             lectura, selección, entrenamiento, evaluación y utilidades
outputs/         resultados agregados; las predicciones por fila no se versionan
reports/         informe, auditoría, guion de exposición y figuras
tests/           pruebas unitarias rápidas y verificaciones de notebooks
data/raw/        dataset local, ignorado por Git
```

## Entregables

- [`proposal.md`](proposal.md): propuesta y resultados de la primera etapa.
- [`reports/informe_final.md`](reports/informe_final.md): informe técnico basado en resultados guardados.
- [`reports/presentacion_esquema.md`](reports/presentacion_esquema.md): guion de presentación.
- [`reports/avance.md`](reports/avance.md): resumen de la etapa intermedia.
- [`reports/auditoria_integracion.md`](reports/auditoria_integracion.md): inventario, discrepancias y riesgos de trazabilidad.
