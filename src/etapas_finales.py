"""Herramientas reproducibles para calibracion temporal y evaluacion final.

Los estimadores, transformaciones y calibradores se ajustan en conjuntos
cronologicos separados. Este modulo solo recibe prestamos con desenlace final
para modelar; `auditar_test` conserva aparte los conteos de estados activos.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from . import features as ft
from .config import MAPA_OBJETIVO, SEMILLA, localizar_datos
from .modelado import TIPOS_AUDITADOS


ANIOS_VALIDACION = [2013, 2014, 2015, 2016]
UMBRAL_FALTANTES = 50
PARAMETROS = {
    "regresion_logistica": {
        "C": 0.03148911647956861,
        "class_weight": None,
        "tol": 8.66211944751964e-05,
    },
    "svm_lineal": {
        "C": 0.010053940194693016,
        "class_weight": "balanced",
        "tol": 0.0001,
    },
    "gradient_boosting": {
        "learning_rate": 0.05279087869966013,
        "max_iter": 200,
        "max_leaf_nodes": 25,
        "min_samples_leaf": 31,
        "l2_regularization": 9.149845592811953,
    },
}


def _columnas_candidatas(ruta):
    fuente = pd.read_csv(ruta, nrows=0).columns.tolist()
    candidatas = ft.seleccionar_predictoras(
        pd.DataFrame(columns=fuente), [], incluir_precio_lc=False
    )
    return fuente, candidatas


def _dtype_lectura(columnas):
    tipos = {}
    for c in columnas:
        sugerido = TIPOS_AUDITADOS.get(c)
        if sugerido is None:
            raise ValueError(f"No existe tipo auditado para {c}.")
        tipos[c] = "string" if sugerido in ["category", "string"] else sugerido
    return tipos


def _preparar_bloque(bloque, mapa_objetivo=MAPA_OBJETIVO):
    issue_original = bloque["issue_d"].notna()
    bloque["issue_d"] = pd.to_datetime(
        bloque["issue_d"], format="%b-%Y", errors="coerce"
    )
    if (issue_original & bloque["issue_d"].isna()).any():
        raise ValueError("Hay fechas issue_d que no se pudieron interpretar.")
    for c in ["earliest_cr_line", "sec_app_earliest_cr_line"]:
        if c in bloque:
            original = bloque[c].notna()
            convertido = pd.to_datetime(bloque[c], format="%b-%Y", errors="coerce")
            if (original & convertido.isna()).any():
                raise ValueError(f"Hay fechas no reconocidas en {c}.")
            bloque[c] = convertido
    if "term" in bloque:
        bloque["term"] = pd.to_numeric(
            bloque["term"].astype("string").str.strip().str.replace(
                " months", "", regex=False
            ), errors="coerce"
        ).astype("float64")
    bloque["fecha_prestamo"] = bloque["issue_d"]
    bloque["anio"] = bloque["issue_d"].dt.year.astype("Int16")
    bloque["default"] = bloque["loan_status"].map(mapa_objetivo).astype("Int8")
    # Cambia fechas de historial por antiguedad relativa al otorgamiento.
    for c in ["earliest_cr_line", "sec_app_earliest_cr_line"]:
        if c in bloque:
            start = bloque[c]
            end = bloque["fecha_prestamo"]
            bloque["meses_desde_" + c] = (
                (end.dt.year - start.dt.year) * 12 + end.dt.month - start.dt.month
            ).astype("float64")
            bloque = bloque.drop(columns=c)
    return bloque


def cargar_desarrollo(ruta=None, ultimo_anio=2016, chunksize=100_000, verbose=True):
    """Lee solo columnas candidatas y prestamos con estado final hasta el año dado."""
    ruta = localizar_datos(ruta) if ruta is not None else localizar_datos()
    _, candidatas = _columnas_candidatas(ruta)
    columnas = list(dict.fromkeys(candidatas + ["loan_status", "issue_d"]))
    tipos = _dtype_lectura(columnas)
    partes = []
    filas_fuente = 0
    for n, bloque in enumerate(pd.read_csv(
        ruta, usecols=columnas, dtype=tipos, chunksize=chunksize,
        skipinitialspace=True, low_memory=False
    ), start=1):
        filas_fuente += len(bloque)
        bloque = _preparar_bloque(bloque)
        elegibles = (
            bloque["loan_status"].isin(MAPA_OBJETIVO)
            & bloque["anio"].notna()
            & (bloque["anio"] <= ultimo_anio)
        )
        bloque = bloque.loc[elegibles].copy()
        if not bloque.empty:
            partes.append(bloque)
        if verbose and n % 10 == 0:
            print(f"Bloques leídos: {n:,}; filas fuente recorridas: {filas_fuente:,}", flush=True)
    if not partes:
        raise ValueError("No se encontraron préstamos elegibles.")
    datos = pd.concat(partes, ignore_index=True)
    for c in datos.select_dtypes(include="string").columns:
        datos[c] = datos[c].astype("category")
    datos["anio"] = datos["anio"].astype("int16")
    datos["default"] = datos["default"].astype("int8")
    return datos, candidatas, ruta


def preparar_X(datos, candidatas):
    # En la lectura ya sustituimos estas fechas por antigüedades relativas.
    mapa_fechas = {
        "earliest_cr_line": "meses_desde_earliest_cr_line",
        "sec_app_earliest_cr_line": "meses_desde_sec_app_earliest_cr_line",
    }
    columnas = [mapa_fechas.get(c, c) for c in candidatas]
    faltantes = [c for c in columnas if c not in datos.columns]
    if faltantes:
        raise KeyError(f"No se encuentran predictoras preparadas: {faltantes}")
    X = datos[columnas].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        else:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def crear_modelo(nombre):
    p = PARAMETROS[nombre]
    if nombre == "regresion_logistica":
        return LogisticRegression(**p, solver="lbfgs", max_iter=3000,
                                  random_state=SEMILLA)
    if nombre == "svm_lineal":
        return LinearSVC(**p, dual="auto", max_iter=7000,
                         random_state=SEMILLA)
    if nombre == "gradient_boosting":
        return HistGradientBoostingClassifier(**p, early_stopping=False,
                                              random_state=SEMILLA)
    raise KeyError(nombre)


def _crear_preprocesador(X_train, denso):
    numericas = X_train.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in X_train.columns if c not in numericas]
    bloques = []
    if numericas:
        bloques.append(("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler(with_mean=denso)),
        ]), numericas))
    if categoricas:
        bloques.append(("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="Sin información")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first",
                                      sparse_output=not denso, dtype=np.float32)),
        ]), categoricas))
    return ColumnTransformer(
        bloques, remainder="drop", sparse_threshold=0.0 if denso else 1.0
    )


def preparar_corte(X, y, anios, anio_base, anio_calibracion, anio_evaluacion,
                   denso=False):
    """Ajusta selector/imputación/codificación en base; devuelve matrices para cada tramo."""
    i_base = np.flatnonzero(anios <= anio_base)
    i_cal = np.flatnonzero(anios == anio_calibracion)
    i_eval = np.flatnonzero(anios == anio_evaluacion)
    if min(len(i_base), len(i_cal), len(i_eval)) == 0:
        raise ValueError("Uno de los periodos temporales no tiene observaciones.")
    assert anio_base < anio_calibracion < anio_evaluacion
    X_base = X.iloc[i_base]
    faltantes = X_base.isna().mean() * 100
    seleccionadas = [c for c in X.columns
                     if faltantes[c] <= UMBRAL_FALTANTES and X_base[c].notna().any()]
    if not seleccionadas:
        raise ValueError("El criterio del 50% dejó cero variables.")
    pre = _crear_preprocesador(X_base[seleccionadas], denso=denso)
    A = pre.fit_transform(X_base[seleccionadas], y[i_base])
    B = pre.transform(X.iloc[i_cal][seleccionadas])
    C = pre.transform(X.iloc[i_eval][seleccionadas])
    if denso:
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        C = np.asarray(C, dtype=np.float32)
    return {
        "idx_base": i_base, "idx_cal": i_cal, "idx_eval": i_eval,
        "X_base": A, "X_cal": B, "X_eval": C,
        "y_base": y[i_base], "y_cal": y[i_cal], "y_eval": y[i_eval],
        "variables": seleccionadas,
        "faltantes": faltantes,
        "transformadas": A.shape[1],
    }


def puntuaciones_modelo(modelo, nombre, X):
    if nombre in ["regresion_logistica", "gradient_boosting"]:
        return np.asarray(modelo.predict_proba(X)[:, 1], dtype="float64"), "probabilidad"
    return np.asarray(modelo.decision_function(X), dtype="float64"), "margen"


def ajustar_calibradores(scores_cal, y_cal, tipo_score):
    """Platt/sigmoide e isotónica; ajusta calibradores solo con scores de calibración."""
    scores_cal = np.asarray(scores_cal, dtype="float64")
    if tipo_score == "probabilidad":
        z = logit(np.clip(scores_cal, 1e-6, 1 - 1e-6))
    else:
        z = scores_cal
    sigmoid = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                  random_state=SEMILLA)
    sigmoid.fit(z.reshape(-1, 1), y_cal)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isotonic.fit(scores_cal, y_cal)
    return sigmoid, isotonic


def predecir_calibrado(sigmoid, isotonic, scores, tipo_score):
    scores = np.asarray(scores, dtype="float64")
    if tipo_score == "probabilidad":
        z = logit(np.clip(scores, 1e-6, 1 - 1e-6))
    else:
        z = scores
    p_sig = sigmoid.predict_proba(z.reshape(-1, 1))[:, 1]
    p_iso = isotonic.predict(scores)
    return np.clip(p_sig, 1e-12, 1 - 1e-12), np.clip(p_iso, 1e-12, 1 - 1e-12)


def calcular_metricas(y, p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-12, 1 - 1e-12)
    return {
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier_Score": float(brier_score_loss(y, p)),
        "Log_Loss": float(log_loss(y, p, labels=[0, 1])),
        "prevalencia": float(np.mean(y)),
        "n": int(len(y)),
    }


def correr_corte(X, y, anios, anio_validacion, verbose=True):
    """Entrena una vez por algoritmo, calibra en el año previo y evalúa después."""
    anio_base = anio_validacion - 2
    anio_cal = anio_validacion - 1
    filas = []
    predicciones = pd.DataFrame({"anio": anio_validacion,
                                 "default": y[anios == anio_validacion]})
    variables_fold = []
    for nombre in PARAMETROS:
        denso = nombre == "gradient_boosting"
        partes = preparar_corte(X, y, anios, anio_base, anio_cal,
                                anio_validacion, denso=denso)
        modelo = crear_modelo(nombre)
        modelo.fit(partes["X_base"], partes["y_base"])
        s_cal, tipo_score = puntuaciones_modelo(modelo, nombre, partes["X_cal"])
        s_eval, _ = puntuaciones_modelo(modelo, nombre, partes["X_eval"])
        sigmoid, isotonic = ajustar_calibradores(s_cal, partes["y_cal"], tipo_score)
        p_sig, p_iso = predecir_calibrado(sigmoid, isotonic, s_eval, tipo_score)
        metodos = {}
        if nombre != "svm_lineal":
            metodos["Original"] = s_eval
        metodos["Sigmoide"] = p_sig
        metodos["Isotónica"] = p_iso
        for metodo, p in metodos.items():
            metricas = calcular_metricas(partes["y_eval"], p)
            filas.append({"modelo": nombre, "metodo": metodo,
                          "anio_calibracion": anio_cal,
                          "anio_evaluacion": anio_validacion,
                          "anio_base_hasta": anio_base,
                          "n_base": len(partes["y_base"]),
                          "n_calibracion": len(partes["y_cal"]),
                          "variables_originales": len(partes["variables"]),
                          "columnas_transformadas": partes["transformadas"],
                          **metricas})
            predicciones[f"{nombre}__{metodo}"] = p
        for c in X.columns:
            variables_fold.append({
                "anio_evaluacion": anio_validacion,
                "anio_base_hasta": anio_base,
                "variable": c,
                "porcentaje_faltante_base": float(partes["faltantes"][c]),
                "seleccionada": c in partes["variables"],
            })
        if verbose:
            print(f"  {nombre}: base hasta {anio_base} (n={len(partes['y_base']):,}), "
                  f"calibración {anio_cal} (n={len(partes['y_cal']):,}), "
                  f"evaluación {anio_validacion} (n={len(partes['y_eval']):,}); "
                  f"variables={len(partes['variables'])}, transformadas={partes['transformadas']}.",
                  flush=True)
        del modelo, partes, sigmoid, isotonic, s_cal, s_eval, p_sig, p_iso
        gc.collect()
    return pd.DataFrame(filas), predicciones, pd.DataFrame(variables_fold)


def auditar_y_cargar_test(ruta=None, chunksize=100_000, verbose=True):
    """Cuenta estados de todos los préstamos >=2017 y carga solo etiquetas finales."""
    ruta = localizar_datos(ruta) if ruta is not None else localizar_datos()
    fuente, candidatas = _columnas_candidatas(ruta)
    columnas = list(dict.fromkeys(candidatas + ["loan_status", "issue_d"]))
    tipos = _dtype_lectura(columnas)
    conteos = {}
    partes = []
    for n, bloque in enumerate(pd.read_csv(
        ruta, usecols=columnas, dtype=tipos, chunksize=chunksize,
        skipinitialspace=True, low_memory=False
    ), start=1):
        issue = pd.to_datetime(bloque["issue_d"], format="%b-%Y", errors="coerce")
        mask = issue.notna() & (issue.dt.year >= 2017)
        if not mask.any():
            continue
        b = bloque.loc[mask].copy()
        b["anio"] = issue.loc[mask].dt.year.astype("int16")
        for (year, status), count in b.groupby(["anio", "loan_status"], dropna=False).size().items():
            key = (int(year), "[VACÍO]" if pd.isna(status) else str(status))
            conteos[key] = conteos.get(key, 0) + int(count)
        final = b["loan_status"].isin(MAPA_OBJETIVO)
        if final.any():
            b = b.loc[final].copy()
            b = _preparar_bloque(b)
            partes.append(b)
        if verbose and n % 10 == 0:
            print(f"Auditoría de test: bloque {n:,} recorrido.", flush=True)
    tabla_conteos = pd.DataFrame([
        {"anio": k[0], "loan_status": k[1], "prestamos": v}
        for k, v in conteos.items()
    ]).sort_values(["anio", "prestamos"], ascending=[True, False])
    if not partes:
        raise ValueError("El periodo de test no contiene préstamos con desenlace final.")
    test = pd.concat(partes, ignore_index=True)
    for c in test.select_dtypes(include="string").columns:
        test[c] = test[c].astype("category")
    test["default"] = test["default"].astype("int8")
    test["anio"] = test["anio"].astype("int16")
    return tabla_conteos, test, candidatas, ruta


def entrenar_final_y_evaluar(datos, candidatas, test, metodo_por_modelo):
    """Refit base <=2015, calibración 2016 y una sola evaluación en test >=2017."""
    anios = datos["anio"].to_numpy()
    y = datos["default"].to_numpy(dtype="int8")
    mascara_base = anios <= 2015
    mascara_cal = anios == 2016
    if not mascara_base.any() or not mascara_cal.any():
        raise ValueError("Falta entrenamiento hasta 2015 o calibración 2016.")
    X_dev = preparar_X(datos, candidatas)
    X_test = preparar_X(test, candidatas)
    X_base = X_dev.loc[mascara_base]
    X_cal = X_dev.loc[mascara_cal]
    y_base = y[mascara_base]
    y_cal = y[mascara_cal]
    y_test = test["default"].to_numpy(dtype="int8")
    anio_cal = test["anio"].to_numpy(dtype="int16")
    filas, pred = [], pd.DataFrame({"anio": anio_cal, "default": y_test})
    for nombre, metodo in metodo_por_modelo.items():
        denso = nombre == "gradient_boosting"
        faltantes = X_base.isna().mean() * 100
        seleccionadas = [c for c in X_base.columns
                         if faltantes[c] <= UMBRAL_FALTANTES and X_base[c].notna().any()]
        pre = _crear_preprocesador(X_base[seleccionadas], denso)
        A = pre.fit_transform(X_base[seleccionadas], y_base)
        B = pre.transform(X_cal[seleccionadas])
        C = pre.transform(X_test[seleccionadas])
        if denso:
            A, B, C = [np.asarray(x, dtype=np.float32) for x in (A, B, C)]
        modelo = crear_modelo(nombre)
        modelo.fit(A, y_base)
        s_cal, tipo_score = puntuaciones_modelo(modelo, nombre, B)
        s_test, _ = puntuaciones_modelo(modelo, nombre, C)
        sigmoide, isotonic = ajustar_calibradores(s_cal, y_cal, tipo_score)
        p_sig, p_iso = predecir_calibrado(sigmoide, isotonic, s_test, tipo_score)
        if metodo == "Original":
            if nombre == "svm_lineal":
                raise ValueError("SVM lineal requiere calibración para probabilidades.")
            p_final = np.clip(s_test, 1e-12, 1 - 1e-12)
        elif metodo == "Sigmoide":
            p_final = p_sig
        elif metodo == "Isotónica":
            p_final = p_iso
        else:
            raise ValueError(f"Método de calibración desconocido: {metodo}")
        m = calcular_metricas(y_test, p_final)
        filas.append({"modelo": nombre, "metodo": metodo,
                      "variables_originales": len(seleccionadas),
                      "columnas_transformadas": A.shape[1],
                      "anio_base_hasta": 2015, "anio_calibracion": 2016,
                      "anios_test": ",".join(map(str, sorted(np.unique(anio_cal)))),
                      **m})
        pred[nombre] = p_final
        del pre, A, B, C, modelo, sigmoide, isotonic, p_sig, p_iso, s_cal, s_test
        gc.collect()
    return pd.DataFrame(filas), pred

