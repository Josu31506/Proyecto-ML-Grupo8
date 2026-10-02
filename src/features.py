"""Seleccion de variables, control de leakage y pipeline de preprocesamiento.

La regla que ordena este modulo: en `X` solo entra informacion que existia en
el momento en que se decide otorgar el prestamo. Todo lo que se genera despues
(pagos, recuperaciones, planes de dificultades, acuerdos de deuda, el ultimo
FICO consultado) queda fuera de forma explicita y documentada.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columnas que existen para construir el objetivo o partir por fecha,
# pero que nunca entran como predictoras.
AUXILIARES = ["loan_status", "default", "issue_d", "fecha_prestamo", "anio"]

IDENTIFICADORES = ["id", "member_id", "url"]

# Texto libre: fuera del alcance del primer modelo (no por ser inutil).
TEXTO_LIBRE = ["desc", "title", "emp_title"]

# Alta cardinalidad / codigo constante.
DESCARTADAS_EXTRA = ["zip_code", "policy_code"]

# Informacion generada DESPUES de otorgar el prestamo: leakage directo.
POSTERIORES = [
    "pymnt_plan", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "deferral_term", "payment_plan_start_date",
    "orig_projected_additional_accrued_interest",
]
PREFIJOS_POSTERIORES = ("hardship_", "debt_settlement_", "settlement_")

# Calificación, condiciones y financiación de LC: escenario adicional.
# Confirmar su disponibilidad para el momento exacto de predicción.
PRECIO_LC = ["int_rate", "grade", "sub_grade", "installment", "funded_amnt",
             "funded_amnt_inv", "initial_list_status", "disbursement_method"]

# Fechas que se transforman en antiguedad (meses) antes de modelar.
FECHAS_A_ANTIGUEDAD = ["earliest_cr_line", "sec_app_earliest_cr_line"]


def columnas_posteriores(columnas):
    """Lista completa de columnas con informacion posterior al otorgamiento."""
    extra = [c for c in columnas if c.startswith(PREFIJOS_POSTERIORES)]
    return [c for c in POSTERIORES + extra if c in columnas]


def tabla_exclusiones(columnas, columnas_vacias):
    """Documenta por que se excluye cada columna. Alimenta el informe."""
    posteriores = set(columnas_posteriores(columnas))
    filas = []
    for columna in columnas:
        if columna in columnas_vacias:
            motivo, grupo = "100% vacia en el archivo completo", "vacia"
        elif columna in IDENTIFICADORES:
            motivo, grupo = "Identificador, sin poder predictivo", "identificador"
        elif columna in TEXTO_LIBRE:
            motivo, grupo = "Texto libre, fuera del alcance del primer modelo", "texto"
        elif columna == "zip_code":
            motivo, grupo = "Alta cardinalidad; se usa addr_state", "cardinalidad"
        elif columna == "policy_code":
            motivo, grupo = "Codigo de politica constante", "constante"
        elif columna in posteriores:
            motivo, grupo = "Informacion posterior al otorgamiento: LEAKAGE", "leakage"
        elif columna in PRECIO_LC:
            motivo, grupo = "Precio/calificacion de Lending Club: se aisla", "precio_lc"
        elif columna in AUXILIARES:
            motivo, grupo = "Auxiliar: objetivo o particion temporal", "auxiliar"
        else:
            continue
        filas.append({"variable": columna, "grupo": grupo, "motivo": motivo})
    return pd.DataFrame(filas)


def seleccionar_predictoras(muestra, columnas_vacias, incluir_precio_lc=False):
    """Devuelve la lista de predictoras segun el escenario elegido."""
    fuera = set(AUXILIARES) | set(IDENTIFICADORES) | set(TEXTO_LIBRE)
    fuera |= set(DESCARTADAS_EXTRA) | set(columnas_vacias)
    fuera |= set(columnas_posteriores(muestra.columns))
    if not incluir_precio_lc:
        fuera |= set(PRECIO_LC)
    return [c for c in muestra.columns if c not in fuera]


def derivar_antiguedades(tabla):
    """Convierte fechas del historial en meses transcurridos hasta el otorgamiento.

    Una fecha absoluta no puede entrar en una regresion logistica, y ademas
    arrastraria la tendencia temporal. La antiguedad en meses si es comparable
    entre prestamos de anios distintos.
    """
    resultado = tabla.copy()
    otorgamiento = resultado["fecha_prestamo"]
    for columna in FECHAS_A_ANTIGUEDAD:
        if columna in resultado.columns:
            inicio = resultado[columna]
            resultado[f"meses_desde_{columna}"] = (
                (otorgamiento.dt.year - inicio.dt.year) * 12
                + (otorgamiento.dt.month - inicio.dt.month)).astype("float64")
            resultado = resultado.drop(columns=[columna])
    return resultado


def normalizar_categoricas(tabla):
    """Pasa las columnas `string` de pandas a `object` con np.nan.

    scikit-learn no sabe evaluar `pd.NA` en un contexto booleano; con
    `object` + `np.nan` el SimpleImputer las trata sin ambiguedad.
    """
    resultado = tabla.copy()
    for columna in resultado.columns:
        if not pd.api.types.is_numeric_dtype(resultado[columna]):
            resultado[columna] = (resultado[columna].astype(object)
                                  .where(resultado[columna].notna(), np.nan))
    return resultado


def preparar_matriz(tabla):
    """Deja la matriz lista para el pipeline: antiguedades + tipos limpios."""
    return normalizar_categoricas(derivar_antiguedades(tabla))


def filtrar_por_faltantes(X_train, predictoras, umbral_excluir, umbral_revision):
    """Aplica las reglas de faltantes usando SOLO el conjunto de entrenamiento."""
    porcentaje = X_train[predictoras].isna().mean() * 100
    excluidas = porcentaje[porcentaje > umbral_excluir].index.tolist()
    conservadas = [c for c in predictoras if c not in excluidas]
    con_indicador = porcentaje[(porcentaje > umbral_revision)
                               & (porcentaje <= umbral_excluir)].index.tolist()
    resumen = pd.DataFrame({
        "porcentaje_faltante_train": porcentaje,
        "decision": np.where(porcentaje > umbral_excluir, "excluir",
                             np.where(porcentaje > umbral_revision,
                                      "imputar + indicador", "imputar")),
    }).sort_values("porcentaje_faltante_train", ascending=False)
    return conservadas, con_indicador, resumen


def construir_preprocesador(X_train, con_indicador):
    """Pipeline de sklearn: imputacion -> codificacion -> escalado.

    Todos los parametros (medianas, categorias, medias y desviaciones) se
    aprenden con `fit` sobre entrenamiento y se reutilizan con `transform`
    en validacion y prueba. Asi no se filtra informacion entre conjuntos.
    """
    numericas = X_train.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in X_train.columns if c not in numericas]

    # Las numericas con muchos faltantes llevan ademas un indicador 0/1 que
    # permite al modelo distinguir un valor observado de uno rellenado.
    num_con_ind = [c for c in numericas if c in con_indicador]
    num_sin_ind = [c for c in numericas if c not in con_indicador]

    paso_con_ind = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    paso_sin_ind = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    paso_cat = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Sin información")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", drop="first",
                                    sparse_output=False)),
    ])

    preprocesador = ColumnTransformer([
        ("num_ind", paso_con_ind, num_con_ind),
        ("num", paso_sin_ind, num_sin_ind),
        ("cat", paso_cat, categoricas),
    ], remainder="drop", verbose_feature_names_out=False)

    return preprocesador, {"numericas_con_indicador": num_con_ind,
                           "numericas": num_sin_ind, "categoricas": categoricas}
