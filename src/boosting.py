"""Gradient Boosting para prediccion de default en Lending Club.

Utiliza HistGradientBoostingClassifier de scikit-learn, que maneja valores
faltantes de forma nativa y no requiere escalado numerico. Se mantiene la
misma particion temporal, muestra, semilla y metricas que el resto del
proyecto.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import evaluate as ev
from . import features as ft
from .config import SEMILLA, UMBRAL_FALTANTES, UMBRAL_REVISION

# Evitar advertencia de loky en Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# Grilla de busqueda: equilibra capacidad predictiva, regularizacion y tiempo de ejecucion.
GRILLA = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 200],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 100],
    "max_depth": [5, 8, None],
}


def construir_preprocesador_boosting(X_train):
    """Preprocesador para Gradient Boosting: imputacion + codificacion, sin escalado.

    Los arboles de boosting no necesitan escalado. Se reutiliza el esquema de
    codificacion One-Hot con min_frequency para controlar la cardinalidad.
    No se aplica drop='first' porque los arboles no requieren eliminar colinealidad.
    """
    numericas = X_train.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in X_train.columns if c not in numericas]

    paso_cat = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Sin información")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                    min_frequency=0.005)),
    ])
    return ColumnTransformer([
        ("num", "passthrough", numericas),
        ("cat", paso_cat, categoricas),
    ], remainder="drop", verbose_feature_names_out=False)


def preparar_datos(particiones, columnas_vacias, incluir_precio_lc=False):
    """Prepara las matrices para boosting, reutilizando la logica comun."""
    candidatas = ft.seleccionar_predictoras(
        particiones["entrenamiento"], columnas_vacias,
        incluir_precio_lc=incluir_precio_lc)
    originales = {
        conjunto: ft.preparar_matriz(df[candidatas + ["fecha_prestamo"]])
            .drop(columns="fecha_prestamo")
        for conjunto, df in particiones.items()
    }
    columnas, _, resumen = ft.filtrar_por_faltantes(
        originales["entrenamiento"], list(originales["entrenamiento"].columns),
        UMBRAL_FALTANTES, UMBRAL_REVISION)

    preprocesador = construir_preprocesador_boosting(
        originales["entrenamiento"][columnas])
    preprocesador.fit(originales["entrenamiento"][columnas])
    nombres = preprocesador.get_feature_names_out()

    X = {c: preprocesador.transform(o[columnas]) for c, o in originales.items()}
    y = {c: particiones[c]["default"].to_numpy() for c in particiones}
    crudos = {c: o[columnas] for c, o in originales.items()}
    return {
        "X": X, "y": y, "crudos": crudos, "preprocesador": preprocesador,
        "nombres": nombres, "columnas": columnas,
        "variables": {
            "predictoras_evaluadas": len(candidatas),
            "predictoras_finales": len(columnas),
            "columnas_modelo": len(nombres),
            "excluidas_por_faltantes": int((resumen["decision"] == "excluir").sum()),
        },
    }


def nuevo_boosting(**parametros):
    """Crea un HistGradientBoostingClassifier con los parametros dados."""
    return HistGradientBoostingClassifier(
        random_state=SEMILLA,
        early_stopping=False,
        **parametros,
    )


def buscar_hiperparametros(datos, grilla=GRILLA, verbose=True):
    """Busqueda exhaustiva moderada; selecciona por ROC-AUC en validacion."""
    X, y = datos["X"], datos["y"]
    filas = []
    combinaciones = list(itertools.product(*grilla.values()))
    inicio = time.time()
    for i, valores in enumerate(combinaciones, 1):
        parametros = dict(zip(grilla.keys(), valores))
        modelo = nuevo_boosting(**parametros)
        modelo.fit(X["entrenamiento"], y["entrenamiento"])
        auc_train = float(roc_auc_score(
            y["entrenamiento"], modelo.predict_proba(X["entrenamiento"])[:, 1]))
        auc_val = float(roc_auc_score(
            y["validacion"], modelo.predict_proba(X["validacion"])[:, 1]))
        filas.append({
            **parametros,
            "auc_entrenamiento": auc_train,
            "auc_validacion": auc_val,
            "n_iteraciones": modelo.n_iter_,
        })
        if verbose and (i % 15 == 0 or i == len(combinaciones)):
            print(f"  {i}/{len(combinaciones)} combinaciones "
                  f"({time.time() - inicio:.0f}s)", flush=True)
    tabla = pd.DataFrame(filas)
    tabla["brecha"] = tabla["auc_entrenamiento"] - tabla["auc_validacion"]
    tabla = tabla.sort_values(
        ["auc_validacion", "brecha"],
        ascending=[False, True]).reset_index(drop=True)
    tabla["max_leaf_nodes"] = tabla["max_leaf_nodes"].astype(object).where(
        tabla["max_leaf_nodes"].notna(), "None")
    tabla["max_depth"] = tabla["max_depth"].astype(object).where(
        tabla["max_depth"].notna(), "None")
    return tabla


def parametros_de_fila(fila):
    """Extrae los hiperparametros de una fila de la tabla de busqueda."""
    val_hojas = fila["max_leaf_nodes"]
    hojas = None if val_hojas == "None" or pd.isna(val_hojas) else int(val_hojas)
    val_prof = fila["max_depth"]
    profundidad = None if val_prof == "None" or pd.isna(val_prof) else int(val_prof)
    return {
        "learning_rate": float(fila["learning_rate"]),
        "max_iter": int(fila["max_iter"]),
        "max_leaf_nodes": hojas,
        "min_samples_leaf": int(fila["min_samples_leaf"]),
        "max_depth": profundidad,
    }


def entrenar_y_evaluar(datos, parametros):
    """Entrena un modelo de boosting y evalua en todos los conjuntos."""
    X, y = datos["X"], datos["y"]
    modelo = nuevo_boosting(**parametros)
    modelo.fit(X["entrenamiento"], y["entrenamiento"])
    scores = {c: modelo.predict_proba(X[c])[:, 1] for c in X}
    umbral = ev.elegir_umbral(y["validacion"], scores["validacion"])
    resultados = {c: ev.evaluar(y[c], scores[c], umbral) for c in X}
    return modelo, scores, resultados


def importancia_por_permutacion(datos, modelo, n_filas=30_000, repeticiones=5):
    """Importancia por permutacion sobre validacion (variables originales)."""
    pipe = Pipeline([("preprocesar", datos["preprocesador"]), ("modelo", modelo)])
    X_val = datos["crudos"]["validacion"]
    y_val = datos["y"]["validacion"]
    if len(X_val) > n_filas:
        indices = np.random.default_rng(SEMILLA).choice(len(X_val), n_filas,
                                                         replace=False)
        X_val, y_val = X_val.iloc[indices], y_val[indices]
    resultado = permutation_importance(pipe, X_val, y_val, scoring="roc_auc",
                                       n_repeats=repeticiones,
                                       random_state=SEMILLA, n_jobs=1)
    return pd.DataFrame({
        "variable": X_val.columns,
        "caida_auc_media": resultado.importances_mean,
        "caida_auc_desv": resultado.importances_std,
    }).sort_values("caida_auc_media", ascending=False).reset_index(drop=True)
