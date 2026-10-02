# Proyecto ML — Grupo 8

Predicción de incumplimiento (`Charged Off`) en préstamos personales de
**Lending Club**, usando únicamente información disponible el día del
otorgamiento.

- **Propuesta completa:** [`proposal.md`](proposal.md)
- **Exploración inicial:** [`notebooks/01_exploracion_inicial.ipynb`](notebooks/01_exploracion_inicial.ipynb)
- **Métricas del baseline:** [`outputs/metrics.json`](outputs/metrics.json)

## Resultado actual

| Modelo | ROC-AUC validación | ROC-AUC prueba | AP validación |
|---|---:|---:|---:|
| logistica_sin_indicadores | 0.7121 | 0.6902 | 0.4016 |
| logistica_con_indicadores | 0.7128 | 0.6922 | 0.4025 |
| logistica_con_precio_lc | 0.7243 | 0.7020 | 0.4125 |
| clase_mayoritaria | 0.5000 | 0.5000 | 0.2134 |

El baseline principal es `logistica_sin_indicadores`. LC se compara con el modelo con indicadores.

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
y colocar `accepted_2007_to_2018Q4.csv.gz` en `data/raw/`.

No hace falta descomprimirlo: el pipeline lee el `.gz` directamente.

Ver [`data/README.md`](data/README.md) para los detalles.

### 3. Ejecutar el pipeline

```bash
python -m src.run_baseline
```

El tiempo depende del equipo y del archivo de entrada. Si el archivo está en otra ruta:

```bash
python -m src.run_baseline --datos /ruta/a/accepted_2007_to_2018Q4.csv.gz
```

Para una corrida rápida con una muestra menor:

```bash
python -m src.run_baseline --muestra 50000
```

### 4. Salidas generadas

```
outputs/
  metrics.json                 métricas de los 4 escenarios × 3 conjuntos
  auditoria.json               cifras del dataset citadas en proposal.md
  coeficientes_baseline.csv    las 20 variables más influyentes
  decisiones_faltantes.csv     qué se hizo con cada variable y por qué
  exclusiones_variables.csv    qué columna se excluyó y con qué motivo
reports/figures/
  01..09_*.png                 las 9 visualizaciones del análisis
```

### 5. Notebook

```bash
jupyter lab notebooks/01_exploracion_inicial.ipynb
```

Ejecutar las celdas en orden. Las secciones 15–17 y el script utilizan
`src/experimentos.py`: sin indicadores, con indicadores, con indicadores + LC
y referencia mayoritaria. Comparten muestra y preparación. Los resultados
se reproducen con el mismo archivo y versiones. El EDA conserva nombres en
español; las matrices finales usan los nombres originales para compartir código.

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
│   └── 01_exploracion_inicial.ipynb exploración + baseline
├── src/
│   ├── config.py                    semilla, rutas, cortes y umbrales
│   ├── data.py                      escaneo del archivo y muestreo
│   ├── features.py                  control de leakage y preprocesamiento
│   ├── train.py                     definición de los modelos
│   ├── evaluate.py                  métricas y selección de umbral
│   └── run_baseline.py              pipeline de extremo a extremo
├── outputs/                         métricas y tablas (regenerables)
└── reports/figures/                 figuras (regenerables)
```

## Reproducibilidad

- Semilla `42` fija en muestreo, partición y modelos (`src/config.py`).
- Versiones fijadas en `requirements.txt`.
- Las cifras de `proposal.md` salen de `outputs/`.
- El CSV original nunca se modifica.
