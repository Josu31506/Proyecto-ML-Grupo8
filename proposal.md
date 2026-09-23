# Propuesta de proyecto — Entrega previa

> Las cifras se generan con `python -m src.run_baseline`. El notebook utiliza las mismas funciones de `src/experimentos.py`. Las métricas de los cuatro escenarios se guardan en `outputs/metrics.json`. Semilla: `42`.

---

## 1. Título del proyecto

**Predicción de incumplimiento en préstamos personales de Lending Club a partir
de información disponible en el momento del otorgamiento.**

---

## 2. Integrantes

Gavino Navarrete Adrian Piero

Hernández Yataco, Josué Renzo

Rojas Cruz, José Alvaro

Sanchez Dominguez, Matias Sebastian

Grupo 8

---

## 3. Dataset elegido

**Opción B del enunciado: Lending Club Loan Data.**

| Característica | Valor |
|---|---|
| Fuente | Kaggle — [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Archivo usado | `accepted_2007_to_2018Q4.csv.gz` |
| Tamaño en disco | 392 MB comprimido (≈ 1.6 GB sin comprimir) |
| Filas | **2 260 701** |
| Columnas | **151** |
| Periodo | Préstamos otorgados entre 2007 y 2018-Q4 |

### Distribución de `loan_status` en el archivo completo

| Estado | Filas | % | ¿Entra al modelo? |
|---|---:|---:|---|
| Fully Paid | 1 076 751 | 47.63 % | **Sí → `default = 0`** |
| Current | 878 317 | 38.85 % | No: sin desenlace final observado |
| Charged Off | 268 559 | 11.88 % | **Sí → `default = 1`** |
| Late (31-120 days) | 21 467 | 0.95 % | No: atraso ≠ incobrable |
| In Grace Period | 8 436 | 0.37 % | No: desenlace pendiente |
| Late (16-30 days) | 4 349 | 0.19 % | No: atraso ≠ incobrable |
| Does not meet the credit policy. Status:Fully Paid | 1 988 | 0.09 % | No: población especial |
| Does not meet the credit policy. Status:Charged Off | 761 | 0.03 % | No: población especial |
| Default | 40 | 0.00 % | No: etiqueta distinta de `Charged Off` |
| *(sin estado)* | 33 | 0.00 % | No: filas de resumen sin datos de préstamo |

**Préstamos elegibles: 1 345 310.** Sobre ellos se toma una muestra aleatoria
sin reemplazo de **220 003 préstamos (16.35 %)**, proporcional por bloques de
50 000 filas. La muestra existe para reducir el uso de memoria y el tiempo de procesamiento; el escaneo de auditoría sí recorre las 2 260 701 filas.

---

## 4. Pregunta predictiva

> **Dado un préstamo que Lending Club acaba de aprobar, y usando únicamente la
> información disponible el día del desembolso, ¿qué probabilidad hay de que
> termine declarado incobrable (`Charged Off`) en lugar de pagarse por completo?**

Es una clasificación binaria supervisada. El objetivo de negocio es ordenar a
los préstamos aprobados por riesgo como apoyo al análisis crediticio,
no emitir un veredicto automático sobre una persona.

---

## 5. Variable objetivo

`default`, construida a partir de `loan_status`:

```python
default = {"Fully Paid": 0, "Charged Off": 1}[loan_status]
```

- Solo se conservan esas dos etiquetas exactas.
- `Current`, `In Grace Period` y los dos estados `Late` se excluyen porque el
  préstamo aún no tiene desenlace: incluirlos obligaría a inventar una etiqueta.
- `Default` (40 filas) se excluye por ser una etiqueta operativa distinta de
  `Charged Off`. Se documenta la decisión en vez de fusionarlas en silencio.
- `loan_status` **no entra nunca como predictora**; existe solo para derivar `y`.

**Prevalencia en la muestra: 19.83 % de incumplimiento.** El problema está
desbalanceado aproximadamente 4:1, lo que condiciona la elección de métricas.

---

## 6. Unidad de predicción

**Un préstamo individual ya aprobado**, identificado por su fila en el archivo
de aceptados. Una fila = un préstamo = una predicción.

---

## 7. Variables disponibles antes de la predicción

De las 151 columnas originales quedan **65 predictoras** tras aplicar las reglas
de exclusión y el umbral de faltantes. Se agrupan así:

| Grupo | Ejemplos | Disponible el día 0 |
|---|---|---|
| Solicitud | `loan_amnt`, `term`, `purpose`, `application_type` | Sí |
| Perfil declarado | `annual_inc`, `emp_length`, `home_ownership`, `verification_status` | Sí |
| Bureau de crédito | `fico_range_low/high`, `dti`, `revol_util`, `open_acc`, `total_acc`, `delinq_2yrs`, `pub_rec` | Sí |
| Historial agregado | `mo_sin_old_rev_tl_op`, `num_actv_bc_tl`, `tot_hi_cred_lim`, `avg_cur_bal` | Sí |
| Geografía | `addr_state` | Sí |
| Derivada | `meses_desde_earliest_cr_line` (antigüedad del historial en meses) | Sí |

### Escenarios comparables

| Escenario | Predictoras antes de codificar | Indicadores | Papel |
|---|---:|---:|---|
| `logistica_sin_indicadores` | 65 | 0 | Baseline principal |
| `logistica_con_indicadores` | 65 | 31 | Medir el aporte de los indicadores |
| `logistica_con_precio_lc` | 73 | 31 | Medir el aporte adicional de las variables de LC |
| `clase_mayoritaria` | — | — | Referencia que siempre predice Fully Paid |

LC añade `grade`, `sub_grade`, `int_rate`, `installment`, `funded_amnt`, `funded_amnt_inv`, `initial_list_status` y `disbursement_method`. Se comparará con el escenario con indicadores para aislar el aporte de esas ocho variables. Debe confirmarse su disponibilidad en el momento exacto de predicción. Todos los escenarios comparten muestra, periodos y reglas de preparación.

### Variables derivadas

`earliest_cr_line` es una fecha absoluta: entraría al modelo arrastrando la
tendencia temporal. Se transforma en `meses_desde_earliest_cr_line`, los meses
entre el inicio del historial crediticio y el otorgamiento, y se descarta la
fecha original.

---

## 8. Riesgos de leakage

Este es el riesgo principal del dataset, y el que el enunciado penaliza más
fuerte. Se excluyen **38 columnas** por contener información generada *después*
de otorgar el préstamo:

| Familia | Columnas | Por qué es leakage |
|---|---|---|
| Pagos recibidos | `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Solo existen porque el préstamo ya avanzó; un préstamo incobrable deja de pagar |
| Saldos vivos | `out_prncp`, `out_prncp_inv` | Reflejan el estado actual de la deuda |
| Cobranza | `recoveries`, `collection_recovery_fee` | **Solo son distintas de cero si ya hubo charge-off: revelan la etiqueta** |
| Última consulta | `last_credit_pull_d` | Puede reflejar consultas posteriores al otorgamiento |
| FICO actualizado | `last_fico_range_high`, `last_fico_range_low` | Es la consulta *más reciente*, posterior al desembolso |
| Dificultades | Columnas `hardship_*`, `deferral_term`, `payment_plan_start_date`, `orig_projected_additional_accrued_interest` | Un plan por dificultades es consecuencia del deterioro |
| Acuerdos de deuda | `debt_settlement_flag*`, 5 columnas `settlement_*` | Solo existen tras el impago |
| Plan de pagos | `pymnt_plan` | Estado operativo posterior |

`recoveries` contiene recuperaciones posteriores al incumplimiento. Incluirla permitiría usar información futura y podría inflar las métricas; no se ha cuantificado aquí ese efecto.

### Otros riesgos controlados

| Riesgo | Control aplicado |
|---|---|
| **Leakage temporal por partición aleatoria** | La partición es por año de otorgamiento, nunca aleatoria (§10) |
| **Leakage por preprocesamiento** | Medianas, categorías, medias y desviaciones se aprenden con `fit()` **solo en entrenamiento** y se aplican con `transform()` al resto. El script agrupa estos pasos en un `Pipeline`; el notebook los aplica explícitamente. Esto facilita el control, pero no garantiza por sí solo ausencia de leakage |
| **Leakage por selección de umbral** | El umbral de corte se elige en validación, nunca en prueba |
| **Identificadores** | `id`, `url`, `member_id` fuera |
| **Texto libre** | `desc`, `title`, `emp_title` fuera del primer modelo (decisión de alcance, no de utilidad) |
| **Alta cardinalidad** | `zip_code` fuera; se conserva `addr_state` |
| **Constante** | `policy_code` toma un único valor (`1.0`) en todo el archivo |

---

## 9. Métrica principal y métrica secundaria

### Por qué no accuracy

La muestra tiene 19.83 % de incumplimiento y validación 21.34 %; predecir siempre «Fully Paid» da **78.66 % de
accuracy en validación** sin detectar un solo default. Es la referencia
`clase_mayoritaria` y está en `outputs/metrics.json` precisamente para hacer
visible esa trampa.

| Métrica | Rol | Justificación |
|---|---|---|
| **ROC-AUC** | **Principal** | Evalúa la capacidad de ordenar incumplidos por encima de pagados considerando distintos umbrales. Las diferencias entre años deben interpretarse según la población evaluada |
| **Average Precision (AP)** | **Secundaria** | Se concentra en la clase minoritaria, que es la que cuesta dinero. Más exigente que ROC-AUC cuando el positivo es raro |
| KS, Brier, recall, precisión, F1, matriz de confusión | Apoyo | KS es estándar en riesgo crediticio; Brier mide el error de las probabilidades estimadas; el resto describe el comportamiento en un umbral concreto |
| Accuracy | Solo contexto | Se reporta **siempre junto a las demás**, nunca sola |

### Resultados del baseline

AP es Average Precision, guardada como `pr_auc` en JSON. Cada regresión elige su umbral maximizando F1 en validación y lo aplica sin reajustar a prueba. El del baseline sin indicadores es **0.1754**.

| Modelo | Conjunto | ROC-AUC | AP | KS | Recall | Precisión | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistica_sin_indicadores | entrenamiento | 0.7025 | 0.3234 | 0.2966 | 0.6172 | 0.2773 | 0.3827 | 0.6641 |
| logistica_sin_indicadores | validacion | 0.7121 | 0.4016 | 0.3071 | 0.6344 | 0.3440 | 0.4461 | 0.6637 |
| logistica_sin_indicadores | prueba | 0.6902 | 0.3584 | 0.2825 | 0.5764 | 0.3348 | 0.4235 | 0.6667 |
| logistica_con_indicadores | entrenamiento | 0.7031 | 0.3241 | 0.2958 | 0.6271 | 0.2758 | 0.3831 | 0.6594 |
| logistica_con_indicadores | validacion | 0.7128 | 0.4025 | 0.3079 | 0.6495 | 0.3403 | 0.4466 | 0.6564 |
| logistica_con_indicadores | prueba | 0.6922 | 0.3610 | 0.2864 | 0.5917 | 0.3330 | 0.4262 | 0.6615 |
| logistica_con_precio_lc | entrenamiento | 0.7103 | 0.3315 | 0.3056 | 0.6326 | 0.2802 | 0.3884 | 0.6639 |
| logistica_con_precio_lc | validacion | 0.7243 | 0.4125 | 0.3284 | 0.6543 | 0.3515 | 0.4573 | 0.6685 |
| logistica_con_precio_lc | prueba | 0.7020 | 0.3698 | 0.3023 | 0.6192 | 0.3369 | 0.4364 | 0.6602 |
| clase_mayoritaria | entrenamiento | 0.5000 | 0.1687 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8313 |
| clase_mayoritaria | validacion | 0.5000 | 0.2134 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7866 |
| clase_mayoritaria | prueba | 0.5000 | 0.2125 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7875 |

### Lectura de los resultados

El baseline obtiene ROC-AUC **0.7121** y AP **0.4016** en validación, frente a 0.5000 y 0.2134 de la referencia mayoritaria. Su accuracy es 0.6637; la referencia alcanza 0.7866 sin detectar incumplimientos.

Detecta **14,811 de 23,347** incumplimientos (63.4%) y clasifica como incumplidos **28,246 de 86,035** préstamos pagados (32.8%). Son predicciones de riesgo, no rechazos automáticos.

Añadir indicadores cambia ROC-AUC en **+0.0007** en validación. Añadir LC al escenario con indicadores cambia ROC-AUC en **+0.0116**. Son resultados iniciales, no pruebas de causalidad ni de ausencia de leakage.

En prueba, el baseline alcanza ROC-AUC 0.6902. Las diferencias temporales y el seguimiento disponible requieren revisión. F1 no representa directamente los costos económicos; se estudiará una función de costo en la entrega final.

---


## 10. Plan de validación

### Partición temporal, no aleatoria

| Conjunto | Años de otorgamiento | Filas | Tasa de default | Uso |
|---|---|---:|---:|---|
| Entrenamiento | 2007 – 2014 | 73 701 | 16.87 % | Ajustar imputaciones, codificación, escalado y coeficientes |
| Validación | 2015 – 2016 | 109 382 | 21.34 % | Comparar modelos, elegir hiperparámetros y el umbral |
| Prueba | 2017 – 2018 | 36 920 | 21.25 % | Evaluación inicial realizada; no usar para elegir mejoras |

Los tres conjuntos son disjuntos y suman exactamente las 220 003 filas
(verificado con `assert` en `src/data.py`).

**Por qué temporal y no aleatoria:** una partición aleatoria pondría préstamos
de 2018 en entrenamiento y de 2012 en prueba. Eso no reproduciría una evaluación sobre préstamos posteriores y podría dar resultados optimistas. La partición por año reproduce el uso real —
aprender del pasado para decidir sobre solicitudes nuevas — y además permite
medir la deriva: la tasa de default pasa de 16.87 % a 21.34 % entre periodos.

**Nota sobre los cortes:** 2015 y 2017 son una propuesta inicial, no un
resultado demostrado. Se revisarán según el seguimiento disponible, no según las métricas. También debe comprobarse si los resultados de entrenamiento ya se conocían en la fecha que se pretende simular.

### Requerimientos

- Semilla `42` fija en muestreo, partición y modelos.
- En el script, el preprocesamiento vive dentro de un `Pipeline`; `fit` solo ve
  entrenamiento.
- Prueba ya fue consultada en la evaluación inicial. Las decisiones posteriores se basarán en entrenamiento y validación, y se documentará esta limitación.
- El umbral se selecciona en validación y se congela antes de evaluar en prueba.

---

## 11. Modelo baseline

**Regresión logística** (`solver="lbfgs"`, `C=1.0`, `max_iter=2000`, semilla 42).

`src/experimentos.py` comparte preparación y entrenamiento entre notebook y script. El preprocesamiento agrupa imputación, One-Hot y escalado mediante pipelines; se ajusta solo con entrenamiento. La logística se ajusta sobre las matrices resultantes.

- Numéricas: mediana y escalado con parámetros de entrenamiento.
- Indicadores: se añaden solo en los dos escenarios que los incluyen, para las 31 variables seleccionadas.
- Categóricas: `Sin información` y One-Hot (`drop="first"`, `handle_unknown="ignore"`), sin agrupar categorías poco frecuentes.
- Fechas: antigüedad crediticia en meses, retirando la fecha original.


### Tratamiento de faltantes

Partiendo de 95 candidatas (tras excluir leakage, identificadores, texto y
precio de LC), el umbral de faltantes deja **65 predictoras finales**:

| Regla (medida **solo en entrenamiento**) | Acción | Variables |
|---|---|---:|
| > 95 % faltante | Excluir de `X` | **30** |
| Numéricas con > 12 % y ≤ 95 % faltante | Mediana; indicador solo en escenarios que lo incluyen | **31** |
| Numéricas con ≤ 12 % faltante | Mediana (sin indicador) | 28 |
| Categóricas | Categoría explícita `"Sin informacion"` | 6 |

Las 30 excluidas están al **100 % faltantes en entrenamiento** (2007-2014), no
apenas por encima del umbral. Son de tres familias: solicitante secundario
(`sec_app_*`), solicitudes conjuntas (`annual_inc_joint`, `dti_joint`,
`revol_bal_joint`, `verification_status_joint`) y cuentas a plazos abiertas
recientemente (`open_il_*`, `open_rv_*`, `il_util`, `all_util`, `inq_fi`,
`total_cu_tl`, `max_bal_bc`).

Los faltantes se concentran en determinados años. Las variables totalmente vacías no pueden aprovecharse en el entrenamiento actual; se pueden revisar los cortes manteniendo una separación temporal. La fecha de incorporación de cada campo requiere confirmación.

En los escenarios con indicadores, se crean para las 31 numéricas seleccionadas, antes de imputar. Distinguen los valores observados de los rellenados, pero también pueden reflejar el periodo de recopilación.

#### Comparación de indicadores ejecutada

Los dos recorridos ejecutan el mismo experimento con la misma codificación. El baseline usa mediana sin indicadores; el segundo escenario añade los 31 indicadores.

| Escenario | ROC-AUC validación | ROC-AUC prueba |
|---|---:|---:|
| logistica_sin_indicadores | 0.7121 | 0.6902 |
| logistica_con_indicadores | 0.7128 | 0.6922 |

La diferencia en validación es +0.0007. La utilidad se decidirá con validación; no se utilizará prueba para elegir el escenario.

### Referencia de clase mayoritaria

`clase_mayoritaria`: predice siempre «Fully Paid». ROC-AUC 0.5000 y accuracy
0.7866. Cualquier modelo que no la supere en ROC-AUC y AP no aporta nada.

### Interpretación del baseline

`outputs/coeficientes_baseline.csv` corresponde al modelo sin indicadores. Los coeficientes de numéricas escaladas se interpretan por desviación estándar; los de categorías respecto a la categoría de referencia. `exp(coeficiente)` multiplica las odds, `p/(1-p)`, no directamente la probabilidad.

No demuestran causalidad, ausencia de leakage ni una clasificación definitiva de importancia. La figura 9 y la sección 17.5 del notebook muestran este mismo baseline.

---


## 12. Riesgos técnicos

### 12.1. Tiempo de seguimiento de los préstamos

Los préstamos duran 36 o 60 meses. El nombre del archivo identifica el periodo de otorgamiento, pero no confirma la fecha de actualización de sus estados.

Un préstamo a 60 meses otorgado en 2016 vencería en 2021, pero puede pagarse anticipadamente. Al conservar solo resultados finales, los años recientes podrían sobrerrepresentar pagos anticipados e incumplimientos tempranos. Las diferencias entre años no prueban por sí solas esta explicación.

**Medidas previstas:**

1. Confirmar la fecha de actualización y el seguimiento disponible.
2. Evaluar préstamos cuyo vencimiento previsto sea anterior a esa fecha.
3. Revisar los cortes según el seguimiento, no según las métricas obtenidas, y comparar los plazos de 36 y 60 meses.
4. Como posible extensión, estudiar métodos que incluyan préstamos todavía activos.

Los resultados describen préstamos aceptados con desenlace conocido; no permiten inferir cómo habrían pagado los rechazados.

### 12.2. Riesgos éticos

La ubicación y los ingresos pueden reflejar diferencias históricas entre grupos. Se investigará el riesgo de resultados desiguales; los coeficientes actuales no demuestran discriminación ni aquí se ha medido una relación con composición racial.

`zip_code` ya se excluyó. Para la entrega final se compararán las métricas por
estado y por tramo de ingreso, y se evaluará entrenar sin `addr_state` para medir
cómo cambia el desempeño. Retirar esa variable no garantiza eliminar sesgos.

El modelo es una herramienta de apoyo, no un veredicto automático sobre una
persona.

---

## 13. Plan de trabajo — semanas restantes

Para continuar el proyecto, seguiremos estos pasos:

1. Revisar el tiempo de seguimiento de los préstamos. Confirmaremos la fecha de actualización de los datos y si los préstamos tuvieron tiempo suficiente para terminar. Con ello definiremos la población y los periodos de entrenamiento, validación y prueba.
2. Mejorar la preparación de los datos. Revisaremos las variables creadas, el tratamiento de valores extremos y la utilidad de los indicadores de faltantes. Los ajustes se aprenderán únicamente con entrenamiento.
3. Entrenar y comparar modelos. Compararemos la regresión logística, Random Forest y Gradient Boosting utilizando los mismos conjuntos y métricas.
4. Ajustar los modelos. Probaremos distintas configuraciones mediante validación temporal dentro del entrenamiento. Revisaremos las probabilidades estimadas y elegiremos el umbral de clasificación con validación.
5. Analizar los errores. Evaluaremos en qué grupos se equivoca más el modelo, considerando año, plazo, estado de residencia e ingresos. También examinaremos qué variables contribuyen a sus predicciones.
6. Realizar la evaluación final. Una vez elegidos el modelo y sus ajustes, lo evaluaremos en el conjunto de prueba definido. Documentaremos que los resultados iniciales de 2017–2018 ya fueron consultados y evitaremos utilizarlos para elegir mejoras.
7. Preparar los entregables. Organizaremos el código y los resultados, actualizaremos las instrucciones de ejecución y elaboraremos el informe y la presentación con los hallazgos y limitaciones.

### Criterio de éxito

Superar de forma consistente al baseline (**ROC-AUC 0.7121 en validación**) con
una mejora explicada y reproducible, manteniendo el control de leakage. Un
modelo complejo no se elegirá solo por ser complejo. Si cambia la población o la partición, se volverá a entrenar el baseline para comparar en igualdad de condiciones.

---

## Reproducción

```bash
pip install -r requirements.txt
python -m src.run_baseline --datos ruta/al/accepted_2007_to_2018Q4.csv.gz
```

Regenera `outputs/metrics.json`, `outputs/auditoria.json` y las 9 figuras de
`reports/figures/` con un tiempo de ejecución que depende del equipo. Las instrucciones completas
están en [`README.md`](README.md).

## Referencias

- Lending Club. *Data Dictionary* (`data/Lending Club Data Dictionary Approved - ES.csv`).
- scikit-learn developers. *Imputation of missing values*.
  <https://scikit-learn.org/stable/modules/impute.html>
- scikit-learn developers. *Common pitfalls and recommended practices — Data leakage*.
  <https://scikit-learn.org/stable/common_pitfalls.html>
