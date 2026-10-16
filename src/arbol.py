import itertools
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from . import evaluate as ev
from . import features as ft
from .config import SEMILLA, UMBRAL_FALTANTES, UMBRAL_REVISION

GRILLA = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None],
    "min_samples_leaf": [1, 20, 100, 300, 1000, 2000, 4000],
    "class_weight": [None, "balanced"],
}


def construir_preprocesador_arbol(X_train):
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

    preprocesador = construir_preprocesador_arbol(originales["entrenamiento"][columnas])
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


def nuevo_arbol(**parametros):
    return DecisionTreeClassifier(random_state=SEMILLA, **parametros)


def buscar_hiperparametros(datos, grilla=GRILLA, verbose=True):
    X, y = datos["X"], datos["y"]
    filas = []
    combinaciones = list(itertools.product(*grilla.values()))
    inicio = time.time()
    for i, valores in enumerate(combinaciones, 1):
        parametros = dict(zip(grilla.keys(), valores))
        arbol = nuevo_arbol(**parametros).fit(X["entrenamiento"], y["entrenamiento"])
        filas.append({
            **parametros,
            "auc_entrenamiento": roc_auc_score(
                y["entrenamiento"], arbol.predict_proba(X["entrenamiento"])[:, 1]),
            "auc_validacion": roc_auc_score(
                y["validacion"], arbol.predict_proba(X["validacion"])[:, 1]),
            "profundidad_real": arbol.get_depth(),
            "hojas": arbol.get_n_leaves(),
        })
        if verbose and i % 20 == 0:
            print(f"  {i}/{len(combinaciones)} combinaciones "
                  f"({time.time() - inicio:.0f}s)", flush=True)
    tabla = pd.DataFrame(filas)
    tabla["brecha"] = tabla["auc_entrenamiento"] - tabla["auc_validacion"]
    tabla["_profundidad_orden"] = tabla["max_depth"].fillna(10_000)
    tabla = tabla.sort_values(
        ["auc_validacion", "hojas", "_profundidad_orden"],
        ascending=[False, True, True]).drop(columns="_profundidad_orden")
    tabla["max_depth"] = tabla["max_depth"].astype(object).where(
        tabla["max_depth"].notna(), "None")
    return tabla.reset_index(drop=True)


def parametros_de_fila(fila):
    profundidad = None if fila["max_depth"] == "None" else int(fila["max_depth"])
    peso = None if pd.isna(fila["class_weight"]) else fila["class_weight"]
    return {"max_depth": profundidad,
            "min_samples_leaf": int(fila["min_samples_leaf"]),
            "class_weight": peso}


def entrenar_y_evaluar(datos, parametros):
    X, y = datos["X"], datos["y"]
    arbol = nuevo_arbol(**parametros).fit(X["entrenamiento"], y["entrenamiento"])
    scores = {c: arbol.predict_proba(X[c])[:, 1] for c in X}
    umbral = ev.elegir_umbral(y["validacion"], scores["validacion"])
    resultados = {c: ev.evaluar(y[c], scores[c], umbral) for c in X}
    return arbol, scores, resultados


def importancia_por_permutacion(datos, arbol, n_filas=30_000, repeticiones=5):
    modelo = Pipeline([("preprocesar", datos["preprocesador"]), ("arbol", arbol)])
    X_val = datos["crudos"]["validacion"]
    y_val = datos["y"]["validacion"]
    if len(X_val) > n_filas:
        indices = np.random.default_rng(SEMILLA).choice(len(X_val), n_filas,
                                                        replace=False)
        X_val, y_val = X_val.iloc[indices], y_val[indices]
    resultado = permutation_importance(modelo, X_val, y_val, scoring="roc_auc",
                                       n_repeats=repeticiones,
                                       random_state=SEMILLA, n_jobs=1)
    return pd.DataFrame({
        "variable": X_val.columns,
        "caida_auc_media": resultado.importances_mean,
        "caida_auc_desv": resultado.importances_std,
    }).sort_values("caida_auc_media", ascending=False).reset_index(drop=True)


def importancia_gini(datos, arbol):
    return pd.DataFrame({
        "columna": datos["nombres"],
        "importancia_gini": arbol.feature_importances_,
    }).sort_values("importancia_gini", ascending=False).reset_index(drop=True)
