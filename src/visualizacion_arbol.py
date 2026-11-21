"""Gráficos del experimento inicial del árbol de decisión."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree


AZUL, ROJO, VERDE, GRIS = "#4C78A8", "#E45756", "#54A24B", "#9AA0A6"


def _guardar(figura, nombre, salida):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / nombre
    figura.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(figura)
    print(f"  figura -> {ruta}")
    return ruta


def _texto(valor):
    if valor is None or valor == "None":
        return "None"
    if isinstance(valor, float) and np.isnan(valor):
        return "None"
    return str(int(valor)) if isinstance(valor, (int, float)) else str(valor)


def fig_complejidad(busqueda, seleccion, auc_baseline, grilla, salida):
    orden = [str(d) for d in grilla["max_depth"] if d is not None] + ["None"]
    etiquetas = [d if d != "None" else "sin\nlímite" for d in orden]
    peso = busqueda["class_weight"].map(_texto)
    figura, ejes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for eje, hoja, titulo in [
            (ejes[0], 1, "Sin mínimo por hoja (min_samples_leaf=1)"),
            (ejes[1], seleccion["min_samples_leaf"],
             f"Con el mínimo elegido (min_samples_leaf={seleccion['min_samples_leaf']})")]:
        curva = busqueda[(busqueda["min_samples_leaf"] == hoja)
                         & (peso == _texto(seleccion["class_weight"]))].copy()
        curva["max_depth"] = curva["max_depth"].map(_texto)
        curva = curva.set_index("max_depth").loc[orden]
        eje.plot(range(len(orden)), curva["auc_entrenamiento"], marker="o",
                 color=AZUL, label="Entrenamiento (2007-2014)")
        eje.plot(range(len(orden)), curva["auc_validacion"], marker="o",
                 color=ROJO, label="Validación (2015-2016)")
        eje.axhline(auc_baseline, color=GRIS, linestyle="--",
                    label=f"Baseline logístico, validación ({auc_baseline:.4f})")
        eje.axhline(0.5, color="black", linestyle=":", linewidth=0.8, label="Azar")
        eje.set_xticks(range(len(orden)), etiquetas)
        eje.set_xlabel("Profundidad máxima del árbol")
        eje.set_title(titulo, fontsize=10)
        eje.grid(axis="y", alpha=0.3)
    ejes[1].axvline(orden.index(_texto(seleccion["max_depth"])), color=VERDE,
                    linestyle=":", label="Árbol elegido")
    ejes[0].set_ylabel("ROC-AUC")
    ejes[1].legend(fontsize=8, loc="lower right")
    figura.suptitle("10. Complejidad del árbol vs. sobreajuste", fontsize=12)
    figura.tight_layout()
    return _guardar(figura, "10_arbol_complejidad.png", salida)


def fig_reglas(arbol, nombres, salida):
    figura, eje = plt.subplots(figsize=(24, 10))
    plot_tree(arbol, max_depth=3, feature_names=list(nombres),
              class_names=["Fully Paid", "Charged Off"], filled=True,
              proportion=True, impurity=False, rounded=True, fontsize=9, ax=eje)
    eje.set_title("11. Primeros 3 niveles del árbol elegido "
                  "(value = proporción [Fully Paid, Charged Off] en el nodo)",
                  fontsize=14)
    return _guardar(figura, "11_arbol_reglas.png", salida)


def fig_importancia(importancia, salida, top=15):
    tabla = importancia.head(top).iloc[::-1]
    figura, eje = plt.subplots(figsize=(8, 6.5))
    eje.barh(tabla["variable"], tabla["caida_auc_media"],
             xerr=tabla["caida_auc_desv"], color=AZUL)
    eje.set_xlabel("Caída del ROC-AUC en validación al desordenar la variable")
    eje.set_title("12. Importancia por permutación del árbol elegido", fontsize=11)
    eje.tick_params(labelsize=8)
    return _guardar(figura, "12_arbol_importancia.png", salida)


def fig_roc(curvas, salida, referencias_auc=None):
    figura, eje = plt.subplots(figsize=(6.5, 6))
    for (nombre, y, puntuacion), color in zip(curvas, [GRIS, AZUL, ROJO]):
        fpr, tpr, _ = roc_curve(y, puntuacion)
        eje.plot(fpr, tpr, color=color,
                 label=f"{nombre} (AUC={roc_auc_score(y, puntuacion):.4f})")
    eje.plot([0, 1], [0, 1], ":", color="black", linewidth=0.8, label="Azar")
    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos")
    eje.set_title("13. Curvas ROC en validación", fontsize=11)
    eje.legend(fontsize=8, loc="lower right")
    if referencias_auc:
        texto = "Referencias guardadas, sin curvas individuales:\n" + "\n".join(
            f"{nombre}: AUC={valor:.4f}" for nombre, valor in referencias_auc.items())
        eje.text(0.02, 0.98, texto, transform=eje.transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "alpha": 0.85})
    figura.tight_layout()
    return _guardar(figura, "13_comparacion_roc.png", salida)
