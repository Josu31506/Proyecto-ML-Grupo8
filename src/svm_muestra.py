"""Experimento complementario del notebook 05; separado del pipeline final."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import (ANIO_CORTE_PRUEBA, ANIO_CORTE_VALIDACION, SEMILLA,
                     TAMANO_BLOQUE, TAMANO_MUESTRA)
from .evaluate import elegir_umbral
from .experimento_inicial import cargar_experimento_inicial, leer_json, validar_referencias
from .experimentos import preparar_experimentos

ESCENARIO = ("logistica_sin_indicadores", False, False)
NOMBRES = ("regresion_logistica", "svm_lineal", "arbol_decision", "gradient_boosting")
COMMIT_ORIGEN = "a189be888955c925eef1473c8e6777f9c6342ee8"
ARCHIVOS_VERIFICABLES = (
    "metricas_svm_muestra.csv", "particiones_svm_muestra.csv", "variables_svm_muestra.csv",
    "columnas_transformadas.csv", "configuracion_svm_muestra.json", "muestra_svm_muestra.json",
)


def modelos_originales():
    return {
        "regresion_logistica": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=SEMILLA),
        "svm_lineal": LinearSVC(
            C=0.010053940194693016, class_weight="balanced", dual="auto",
            max_iter=7000, random_state=SEMILLA),
        "arbol_decision": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=1000, random_state=SEMILLA),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=75, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, early_stopping=False, random_state=SEMILLA),
    }


def configuracion_experimento():
    return {
        "commit_origen": COMMIT_ORIGEN,
        "notebook_origen": "notebooks/11_svm_muestra.ipynb",
        "notebook_actual": "notebooks/05_svm_muestra.ipynb",
        "semilla": SEMILLA, "bloque": TAMANO_BLOQUE,
        "tamano_muestra_objetivo": TAMANO_MUESTRA, "filas_muestra_esperadas": 220003,
        "cortes": {"validacion": ANIO_CORTE_VALIDACION, "prueba": ANIO_CORTE_PRUEBA},
        "preprocesamiento": {
            "escenario": ESCENARIO[0], "indicadores_faltantes": False, "precio_lc": False,
            "umbral_excluir_faltantes_pct": 95, "fit_solo_entrenamiento": True,
            "numericas": "mediana y StandardScaler",
            "categoricas": "Sin información y OneHotEncoder(drop='first', handle_unknown='ignore')",
            "matriz_compartida": True,
        },
        "modelos": {n: m.get_params(deep=False) for n, m in modelos_originales().items()},
        "umbral": "Máximo F1 en validación; congelado para entrenamiento y prueba",
        "score_svm": "decision_function; margen sin calibrar",
        "uso_test": "Comparación retrospectiva; no selecciona modelos, parámetros ni umbrales",
    }


def cargar_muestra(ruta_datos=None, raiz=None):
    if (SEMILLA, TAMANO_MUESTRA, TAMANO_BLOQUE, ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA) != (42, 220000, 50000, 2015, 2017):
        raise ValueError("La configuración compartida difiere del experimento SVM original.")
    experimento = cargar_experimento_inicial(ruta_datos=ruta_datos, modo="reproduccion", raiz=raiz, verbose=True)
    if len(experimento["muestra"]) != 220003 or set(experimento["muestra"].default.unique()) != {0, 1}:
        raise ValueError("La muestra no contiene los 220,003 préstamos y ambas clases esperadas.")
    referencia = leer_json(experimento["raiz"] / "outputs/metrics.json")
    validar_referencias(experimento, [referencia])
    return experimento


def preparar_variables(experimento):
    preparados = preparar_experimentos(
        experimento["particiones"], experimento["columnas_vacias"], escenarios=[ESCENARIO])
    datos = preparados[ESCENARIO[0]]
    for conjunto, matriz in datos["X"].items():
        if not np.isfinite(matriz.to_numpy()).all() or not matriz.index.equals(datos["y"][conjunto].index):
            raise ValueError(f"Matriz no finita o etiquetas desalineadas en {conjunto}.")
    return datos


def resumen_particiones(experimento):
    return pd.DataFrame([
        {"conjunto": nombre, "filas": len(tabla), "anio_min": int(tabla.anio.min()),
         "anio_max": int(tabla.anio.max()), "default_pct": float(100 * tabla.default.mean())}
        for nombre, tabla in experimento["particiones"].items()
    ])


def puntuaciones(modelo, matriz):
    if isinstance(modelo, LinearSVC):
        return modelo.decision_function(matriz)
    return modelo.predict_proba(matriz)[:, 1]


def metricas_modelo(nombre, scores, y, segundos):
    umbral = elegir_umbral(y["validacion"], scores["validacion"])
    filas = []
    for conjunto in ("entrenamiento", "validacion", "prueba"):
        objetivo = np.asarray(y[conjunto], dtype="int8")
        score = np.asarray(scores[conjunto], dtype="float64")
        pred = (score >= umbral).astype("int8")
        vn, fp, fn, vp = confusion_matrix(objetivo, pred, labels=[0, 1]).ravel()
        filas.append({
            "modelo": nombre, "conjunto": conjunto, "n": len(objetivo),
            "ROC_AUC": roc_auc_score(objetivo, score), "AP": average_precision_score(objetivo, score),
            "umbral_validacion": umbral,
            "precision": precision_score(objetivo, pred, zero_division=0),
            "recall": recall_score(objetivo, pred, zero_division=0),
            "F1": f1_score(objetivo, pred, zero_division=0), "accuracy": accuracy_score(objetivo, pred),
            "VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp),
            "segundos_entrenamiento": segundos,
        })
    return filas


def entrenar_modelos(datos):
    modelos = modelos_originales()
    y = {n: v.to_numpy(dtype="int8") for n, v in datos["y"].items()}
    scores, metricas = {}, []
    for nombre, modelo in modelos.items():
        inicio = time.perf_counter()
        modelo.fit(datos["X"]["entrenamiento"], y["entrenamiento"])
        segundos = time.perf_counter() - inicio
        scores[nombre] = {n: puntuaciones(modelo, datos["X"][n]) for n in y}
        metricas.extend(metricas_modelo(nombre, scores[nombre], y, segundos))
    return pd.DataFrame(metricas), modelos, scores, y


def guardar_resultados(salida, experimento, datos, metricas):
    salida = Path(salida)
    nombres = [*ARCHIVOS_VERIFICABLES, "decisiones_faltantes.csv"]
    ocupados = [str(salida / n) for n in nombres if (salida / n).exists()]
    if ocupados:
        raise FileExistsError("No se sobrescriben resultados: " + ", ".join(ocupados))
    muestra = {
        "filas_muestra": len(experimento["muestra"]), "semilla": SEMILLA,
        "filas_fuente": int(experimento["auditoria"]["total_filas"]),
        "total_elegibles": int(experimento["auditoria"]["total_elegibles"]),
        "fraccion_muestreo": float(experimento["auditoria"]["fraccion_muestreo"]),
        "columnas_vacias": experimento["columnas_vacias"], "huellas_ids_por_particion": {},
    }
    for n, tabla in experimento["particiones"].items():
        if "id" in tabla:
            ids = tabla.id.astype("string").to_json(orient="values", force_ascii=False)
            muestra["huellas_ids_por_particion"][n] = hashlib.sha256(ids.encode("utf-8")).hexdigest()
    with (salida / "metricas_svm_muestra.csv").open("x", encoding="utf-8", newline="") as f:
        metricas.to_csv(f, index=False)
    resumen_particiones(experimento).to_csv(salida / "particiones_svm_muestra.csv", index=False)
    seleccionadas = datos["faltantes"].loc[datos["faltantes"].decision != "excluir"].copy()
    seleccionadas.index.name = "variable"
    seleccionadas.to_csv(salida / "variables_svm_muestra.csv")
    pd.DataFrame({"variable": datos["X"]["entrenamiento"].columns}).to_csv(salida / "columnas_transformadas.csv", index=False)
    datos["faltantes"].to_csv(salida / "decisiones_faltantes.csv", index_label="variable")
    for nombre, contenido in [("configuracion_svm_muestra.json", configuracion_experimento()),
                              ("muestra_svm_muestra.json", muestra)]:
        with (salida / nombre).open("x", encoding="utf-8") as f:
            json.dump(contenido, f, ensure_ascii=False, indent=2)


def graficar_curvas(scores, y, salida):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nombre, score in scores.items():
        fpr, tpr, _ = roc_curve(y["prueba"], score["prueba"])
        precision, recall, _ = precision_recall_curve(y["prueba"], score["prueba"])
        auc = roc_auc_score(y["prueba"], score["prueba"])
        ap = average_precision_score(y["prueba"], score["prueba"])
        axes[0].plot(fpr, tpr, label=f"{nombre} (AUC={auc:.4f})")
        axes[1].plot(recall, precision, label=f"{nombre} (AP={ap:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Tasa de falsos positivos", ylabel="Recall", title="ROC en prueba")
    axes[1].set(xlabel="Recall", ylabel="Precisión", title="Precision–Recall en prueba")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    archivo = Path(salida) / "curvas_svm_muestra.png"
    if archivo.exists():
        plt.close(fig)
        raise FileExistsError(archivo)
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    return fig
