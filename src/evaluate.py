"""Metricas y curvas de evaluacion.

Metrica principal: ROC-AUC (independiente del umbral, comparable con el Gini
que se usa en scoring crediticio). Metrica secundaria: PR-AUC (average
precision), que es mas informativa que ROC-AUC cuando la clase positiva es
minoritaria. La exactitud se reporta solo como referencia, nunca sola.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def estadistico_ks(y_true, y_score):
    """Maxima separacion entre las acumuladas de buenos y malos."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def elegir_umbral(y_true, y_score, criterio="f1"):
    """Elige el umbral de corte usando SOLO validacion."""
    precision, recall, umbrales = precision_recall_curve(y_true, y_score)
    # precision_recall_curve devuelve un punto mas que umbrales.
    precision, recall = precision[:-1], recall[:-1]
    if criterio == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            puntaje = np.nan_to_num(2 * precision * recall / (precision + recall))
    else:
        raise ValueError(f"Criterio no soportado: {criterio}")
    return float(umbrales[int(np.argmax(puntaje))])


def evaluar(y_true, y_score, umbral):
    """Diccionario de metricas para un conjunto y un umbral dados."""
    y_pred = (y_score >= umbral).astype(int)
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    vn, fp, fn, vp = matriz.ravel()
    return {
        "n": int(len(y_true)),
        "tasa_default_real": float(np.mean(y_true)),
        "umbral": float(umbral),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": estadistico_ks(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "matriz_confusion": {"vn": int(vn), "fp": int(fp),
                             "fn": int(fn), "vp": int(vp)},
    }


def tabla_metricas(resultados):
    """Convierte el diccionario de resultados en una tabla legible."""
    filas = []
    for nombre, conjuntos in resultados.items():
        for conjunto, m in conjuntos.items():
            filas.append({
                "modelo": nombre, "conjunto": conjunto, "n": m["n"],
                "ROC-AUC": round(m["roc_auc"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "KS": round(m["ks"], 4),
                "Recall": round(m["recall"], 4),
                "Precision": round(m["precision"], 4),
                "F1": round(m["f1"], 4),
                "Accuracy": round(m["accuracy"], 4),
            })
    return pd.DataFrame(filas)


def guardar_metricas(resultados, ruta, extra=None):
    """Persiste las metricas en JSON para que el informe no dependa de la sesion."""
    contenido = {"resultados": resultados}
    if extra:
        contenido.update(extra)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return ruta
