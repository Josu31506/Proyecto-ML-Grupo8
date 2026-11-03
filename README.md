# Proyecto ML — Grupo 8

Predicción de incumplimiento (`Charged Off`) en préstamos personales de
**Lending Club**, usando únicamente información disponible el día del
otorgamiento.

- **Propuesta completa:** [`proposal.md`](proposal.md)
- **Exploración inicial:** [`notebooks/01_exploracion_inicial.ipynb`](notebooks/01_exploracion_inicial.ipynb)
- **Árbol de decisión:** [`notebooks/03_arbol_decision.ipynb`](notebooks/03_arbol_decision.ipynb)
- **Gradient Boosting:** [`notebooks/04_gradient_boosting.ipynb`](notebooks/04_gradient_boosting.ipynb)
- **Métricas del baseline:** [`outputs/metrics.json`](outputs/metrics.json)
- **Métricas del árbol:** [`outputs/metrics_arbol.json`](outputs/metrics_arbol.json)
- **Métricas de boosting:** [`outputs/metrics_boosting.json`](outputs/metrics_boosting.json)

## Resultado actual

| Modelo | ROC-AUC validación | ROC-AUC prueba | AP validación |
|---|---:|---:|---:|
| logistica_sin_indicadores (baseline) | 0.7121 | 0.6902 | 0.4016 |
| logistica_con_indicadores | 0.7128 | 0.6922 | 0.4025 |
| logistica_con_precio_lc | 0.7243 | 0.7020 | 0.4125 |
| arbol_elegido (DecisionTree) | 0.6745 | 0.6346 | 0.3548 |
| gradient_boosting_base (HistGradientBoosting) | 0.7007 | 0.6993 | 0.3768 |
| gradient_boosting_con_precio_lc | 0.7157 | 0.7115 | 0.3955 |
| clase_mayoritaria (referencia) | 0.5000 | 0.5000 | 0.2134 |

El baseline principal es `logistica_sin_indicadores`. Los escenarios con precio LC se evalúan de forma independiente.

---

## Reproducir desde cero

### 1. Requisitos

Python 3.11 o superior.

```bash
git clone https://github.com/Josu31506/Proyecto-ML-Grupo8.git
cd Proyecto-ML-Grupo8
python -m venv .venv
```

Activar el entorno:

```bash
source .venv/bin/activate
```

En Windows (PowerShell) es `.venv\Scripts\Activate.ps1`.

```bash
pip install -r requirements.txt
```

### 2. Descargar el dataset

El CSV **no está en el repositorio** (pesa 392 MB comprimido). Descargarlo de
[Kaggle — wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
y colocar `accepted_2007_to_2018Q4.csv.gz` en `data/raw/` o en `~/Downloads/`.

No hace falta descomprimirlo: el pipeline lee el `.gz` directamente.

Ver [`data/README.md`](data/README.md) para los detalles.

### 3. Ejecutar los pipelines

**Baseline (Regresión logística):**
```bash
python -m src.run_baseline
```

**Árbol de decisión:**
```bash
python -m src.run_arbol
```

**Gradient Boosting:**
```bash
python -m src.run_boosting
```

Para una corrida rápida con una muestra menor:

```bash
python -m src.run_boosting --muestra 50000
```

### 4. Salidas generadas

```
outputs/
  metrics.json                 métricas de regresión logística
  metrics_arbol.json           métricas del árbol de decisión
  metrics_boosting.json        métricas de gradient boosting
  auditoria.json               cifras del dataset citadas en proposal.md
  coeficientes_baseline.csv    las 20 variables más influyentes del baseline
  busqueda_arbol.csv           grilla de hiperparámetros del árbol
  busqueda_boosting.csv        grilla de hiperparámetros de boosting
  importancia_arbol.csv        importancia por permutación del árbol
  importancia_boosting.csv     importancia por permutación de boosting
reports/figures/
  01..09_*.png                 visualizaciones del análisis exploratorio y baseline
  10..13_*.png                 visualizaciones del árbol de decisión
  14..16_*.png                 visualizaciones de gradient boosting
```

### 5. Notebooks

```bash
jupyter lab
```

- `notebooks/01_exploracion_inicial.ipynb`: Carga, EDA, control de fuga y baseline.
- `notebooks/03_arbol_decision.ipynb`: Árbol de decisión, poda y reglas.
- `notebooks/04_gradient_boosting.ipynb`: Gradient Boosting, grilla de hiperparámetros e importancia por permutación.

---

## Estructura

```
Proyecto-ML-Grupo8/
├── README.md                        este archivo
├── proposal.md                      propuesta (13 secciones del enunciado)
├── requirements.txt                 dependencias con versiones verificadas
├── data/
│   ├── README.md                    cómo obtener el dataset
│   └── raw/                         el CSV va aquí (ignorado por git)
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb exploración + baseline
│   ├── 03_arbol_decision.ipynb      árbol de decisión
│   └── 04_gradient_boosting.ipynb   gradient boosting
├── src/
│   ├── config.py                    semilla, rutas, cortes y umbrales
│   ├── data.py                      escaneo del archivo y muestreo
│   ├── features.py                  control de leakage y preprocesamiento
│   ├── train.py                     constructores de modelos
│   ├── evaluate.py                  métricas y selección de umbral
│   ├── experimentos.py              escenarios compartidos
│   ├── arbol.py                     lógica y búsqueda de árbol
│   ├── run_arbol.py                 pipeline del árbol de decisión
│   ├── boosting.py                  lógica y búsqueda de gradient boosting
│   ├── run_boosting.py              pipeline de gradient boosting
│   └── run_baseline.py              pipeline de extremo a extremo
├── outputs/                         métricas y tablas (regenerables)
└── reports/figures/                 figuras (regenerables)
```

## Reproducibilidad

- Semilla `42` fija en muestreo, partición y modelos (`src/config.py`).
- Versiones fijadas en `requirements.txt`.
- Las cifras de `proposal.md` salen de `outputs/`.
- El CSV original nunca se modifica.
