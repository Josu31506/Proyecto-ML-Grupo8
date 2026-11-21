"""Gráficos del experimento inicial de Gradient Boosting."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


AZUL, ROJO, VERDE, GRIS, NARANJA = (
    "#4C78A8", "#E45756", "#54A24B", "#9AA0A6", "#F58518"
)


def _guardar(figura, nombre, salida):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / nombre
    figura.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(figura)
    print(f"  figura -> {ruta}")
    return ruta


def fig_busqueda(busqueda, auc_baseline, salida):
    """Muestra rendimiento de validación y diferencia frente a entrenamiento."""
    figura, ejes = plt.subplots(1, 2, figsize=(14, 5))
    colores_nodos = {15: AZUL, 31: VERDE, 63: NARANJA}
    for nodos in [15, 31, 63]:
        sub = busqueda[busqueda["max_leaf_nodes"] == nodos]
        if len(sub):
            ejes[0].scatter(sub["learning_rate"], sub["auc_validacion"],
                            label=f"max_leaf_nodes={nodos}",
                            color=colores_nodos.get(nodos, AZUL), alpha=0.6, s=35)
    ejes[0].axhline(auc_baseline, color=GRIS, linestyle="--",
                    label=f"Baseline logístico ({auc_baseline:.4f})")
    ejes[0].set_xlabel("Learning rate")
    ejes[0].set_ylabel("ROC-AUC validación")
    ejes[0].set_title("Rendimiento por learning rate y número de hojas")
    ejes[0].legend(fontsize=8, loc="lower right")
    ejes[0].grid(axis="y", alpha=0.3)

    ejes[1].scatter(busqueda["brecha"], busqueda["auc_validacion"],
                    alpha=0.45, s=25, color=AZUL, label="Configuraciones exploradas")
    mejor = busqueda.iloc[0]
    ejes[1].scatter([mejor["brecha"]], [mejor["auc_validacion"]],
                    color=ROJO, s=90, zorder=5, label="Modelo elegido")
    ejes[1].axhline(auc_baseline, color=GRIS, linestyle="--",
                    label=f"Baseline logístico ({auc_baseline:.4f})")
    ejes[1].set_xlabel("Brecha (AUC entrenamiento − AUC validación)")
    ejes[1].set_ylabel("ROC-AUC validación")
    ejes[1].set_title("Rendimiento vs. sobreajuste")
    ejes[1].legend(fontsize=8, loc="lower right")
    ejes[1].grid(alpha=0.3)

    figura.suptitle("14. Búsqueda de hiperparámetros — Gradient Boosting", fontsize=12)
    figura.tight_layout()
    return _guardar(figura, "14_boosting_busqueda.png", salida)


def fig_importancia(importancia, salida, top=15):
    """Gráfico de las variables con mayor importancia por permutación."""
    tabla = importancia.head(top).iloc[::-1]
    figura, eje = plt.subplots(figsize=(8, 6.5))
    eje.barh(tabla["variable"], tabla["caida_auc_media"],
             xerr=tabla["caida_auc_desv"], color=VERDE)
    eje.set_xlabel("Caída del ROC-AUC en validación al desordenar la variable")
    eje.set_title("15. Importancia por permutación — Gradient Boosting elegido",
                  fontsize=11)
    eje.tick_params(labelsize=8)
    return _guardar(figura, "15_boosting_importancia.png", salida)


def fig_roc_comparacion(curvas, salida, referencias_auc=None):
    """Curvas ROC de validación y AUC de referencias sin puntuaciones individuales."""
    figura, eje = plt.subplots(figsize=(7, 6))
    colores = [GRIS, AZUL, NARANJA, VERDE, ROJO]
    for i, (nombre, y, puntuacion) in enumerate(curvas):
        fpr, tpr, _ = roc_curve(y, puntuacion)
        color = colores[i % len(colores)]
        eje.plot(fpr, tpr, color=color,
                 label=f"{nombre} (AUC={roc_auc_score(y, puntuacion):.4f})")
    eje.plot([0, 1], [0, 1], ":", color="black", linewidth=0.8, label="Azar")
    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos")
    eje.set_title("16. Curvas ROC en validación — Comparación de modelos", fontsize=11)
    eje.legend(fontsize=8, loc="lower right")
    if referencias_auc:
        texto = "AUC de referencia (curvas individuales omitidas):\n" + "\n".join(
            f"{nombre}: AUC={valor:.4f}" for nombre, valor in referencias_auc.items())
        eje.text(0.02, 0.98, texto, transform=eje.transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "alpha": 0.85})
    figura.tight_layout()
    return _guardar(figura, "16_boosting_comparacion_roc.png", salida)
