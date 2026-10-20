"""Pipeline reproducible para Gradient Boosting.

Ejecutar desde la raiz del proyecto:

    python -m src.run_boosting

Produce:
  outputs/metrics_boosting.json     metricas de todos los escenarios
  outputs/busqueda_boosting.csv     resultados de la busqueda de hiperparametros
  outputs/importancia_boosting.csv  importancia por permutacion
  reports/figures/14..16_*.png      visualizaciones del analisis
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from . import boosting as gb
from . import evaluate as ev
from .config import (ANIO_CORTE_PRUEBA, ANIO_CORTE_VALIDACION, DIR_FIGURAS,
                     DIR_SALIDAS, SEMILLA, TAMANO_MUESTRA, localizar_datos)
from .data import dividir_por_anio, escanear_y_muestrear
from .experimentos import entrenar_experimentos, preparar_experimentos

AZUL, ROJO, VERDE, GRIS, NARANJA = "#4C78A8", "#E45756", "#54A24B", "#9AA0A6", "#F58518"
BASELINE = "logistica_sin_indicadores"


def _guardar(fig, nombre):
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / nombre, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {nombre}")


def fig_busqueda(busqueda, auc_baseline):
    """Muestra la relacion entre ROC-AUC de validacion y la brecha train-val."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    colores_nodos = {15: AZUL, 31: VERDE, 63: NARANJA}
    for nodos in [15, 31, 63]:
        sub = busqueda[busqueda["max_leaf_nodes"] == nodos]
        if len(sub):
            ejes[0].scatter(sub["learning_rate"], sub["auc_validacion"],
                            label=f"max_leaf_nodes={nodos}",
                            color=colores_nodos.get(nodos, AZUL),
                            alpha=0.6, s=35)
    ejes[0].axhline(auc_baseline, color=GRIS, linestyle="--",
                    label=f"Baseline logístico ({auc_baseline:.4f})")
    ejes[0].set_xlabel("Learning rate")
    ejes[0].set_ylabel("ROC-AUC validación")
    ejes[0].set_title("Rendimiento por learning rate y número de hojas")
    ejes[0].legend(fontsize=8, loc="lower right")
    ejes[0].grid(axis="y", alpha=0.3)

    # Panel derecho: AUC validacion vs brecha (train - val)
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

    fig.suptitle("14. Búsqueda de hiperparámetros — Gradient Boosting", fontsize=12)
    fig.tight_layout()
    _guardar(fig, "14_boosting_busqueda.png")


def fig_importancia(importancia, top=15):
    """Top variables por importancia de permutacion."""
    tabla = importancia.head(top).iloc[::-1]
    fig, eje = plt.subplots(figsize=(8, 6.5))
    eje.barh(tabla["variable"], tabla["caida_auc_media"],
             xerr=tabla["caida_auc_desv"], color=VERDE)
    eje.set_xlabel("Caída del ROC-AUC en validación al desordenar la variable")
    eje.set_title("15. Importancia por permutación — Gradient Boosting elegido",
                  fontsize=11)
    eje.tick_params(labelsize=8)
    _guardar(fig, "15_boosting_importancia.png")


def fig_roc_comparacion(curvas):
    """Curvas ROC comparativas en validacion."""
    fig, eje = plt.subplots(figsize=(7, 6))
    colores = [GRIS, AZUL, NARANJA, VERDE, ROJO]
    for i, (nombre, y, s) in enumerate(curvas):
        fpr, tpr, _ = roc_curve(y, s)
        color = colores[i % len(colores)]
        eje.plot(fpr, tpr, color=color,
                 label=f"{nombre} (AUC={roc_auc_score(y, s):.4f})")
    eje.plot([0, 1], [0, 1], ":", color="black", linewidth=0.8, label="Azar")
    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos")
    eje.set_title("16. Curvas ROC en validación — Comparación de modelos", fontsize=11)
    eje.legend(fontsize=8, loc="lower right")
    _guardar(fig, "16_boosting_comparacion_roc.png")


def main(ruta_datos=None, tamano_muestra=TAMANO_MUESTRA):
    inicio = time.time()
    ruta = localizar_datos(ruta_datos)
    print(f"Dataset: {ruta}\n")

    print("[1/6] Cargando y muestreando (mismo procedimiento que el baseline)...")
    muestra, auditoria = escanear_y_muestrear(ruta, tamano_muestra=tamano_muestra,
                                              verbose=False)
    entrenamiento, validacion, prueba = dividir_por_anio(
        muestra, ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA)
    particiones = {"entrenamiento": entrenamiento, "validacion": validacion,
                   "prueba": prueba}
    faltantes = auditoria["faltantes_globales"]
    columnas_vacias = faltantes[faltantes == auditoria["total_filas"]].index.tolist()
    for nombre, parte in particiones.items():
        print(f"  {nombre:<14} {len(parte):>8,} filas  "
              f"default={parte['default'].mean():.4f}")

    print("\n[2/6] Reentrenando el baseline logístico para comparar en igualdad...")
    preparados = preparar_experimentos(particiones, columnas_vacias)
    res_logistica, _, scores_logistica = entrenar_experimentos(preparados)
    auc_baseline = res_logistica[BASELINE]["validacion"]["roc_auc"]
    print(f"  baseline ROC-AUC validación: {auc_baseline:.4f}")

    print("\n[3/6] Preparando datos para Gradient Boosting...")
    datos = gb.preparar_datos(particiones, columnas_vacias, incluir_precio_lc=False)
    print(f"  {datos['variables']['predictoras_finales']} predictoras -> "
          f"{datos['variables']['columnas_modelo']} columnas tras codificación")

    n_combinaciones = int(np.prod([len(v) for v in gb.GRILLA.values()]))
    print(f"\n[4/6] Búsqueda de hiperparámetros ({n_combinaciones} combinaciones, "
          "se elige por ROC-AUC en validación)...")
    busqueda = gb.buscar_hiperparametros(datos)
    mejor = busqueda.iloc[0]
    parametros = gb.parametros_de_fila(mejor)
    print(f"  elegido: {parametros}")
    print(f"  ROC-AUC entrenamiento={mejor['auc_entrenamiento']:.4f}  "
          f"validación={mejor['auc_validacion']:.4f}  "
          f"brecha={mejor['brecha']:.4f}")

    print("\n[5/6] Entrenando el modelo elegido y escenarios adicionales...")
    modelo, scores, res_boosting = gb.entrenar_y_evaluar(datos, parametros)

    # Escenario con precio LC.
    datos_precio = gb.preparar_datos(particiones, columnas_vacias,
                                     incluir_precio_lc=True)
    _, _, res_precio = gb.entrenar_y_evaluar(datos_precio, parametros)

    importancia = gb.importancia_por_permutacion(datos, modelo)

    print("\n[6/6] Figuras y resultados...")
    fig_busqueda(busqueda, auc_baseline)
    fig_importancia(importancia)

    # Curvas ROC comparativas: baseline logistico, arbol elegido, boosting elegido
    curvas_roc = [
        ("Baseline logístico", *scores_logistica[BASELINE]),
        ("Gradient Boosting elegido", datos["y"]["validacion"], scores["validacion"]),
    ]

    try:
        from . import arbol as ar
        datos_arbol = ar.preparar_datos(particiones, columnas_vacias,
                                        incluir_precio_lc=False)
        arbol_json = json.loads(
            (DIR_SALIDAS / "metrics_arbol.json").read_text(encoding="utf-8"))
        params_arbol = {k: (None if v == "None" else int(v) if k != "class_weight" else v)
                        for k, v in arbol_json["parametros_elegidos"].items()}
        arbol_modelo, arbol_scores, _ = ar.entrenar_y_evaluar(datos_arbol, params_arbol)
        curvas_roc.insert(1, ("Árbol de decisión elegido",
                              datos_arbol["y"]["validacion"],
                              arbol_scores["validacion"]))
    except Exception as e:
        print(f"  (Árbol no incluido en curva ROC comparativa: {e})")

    fig_roc_comparacion(curvas_roc)

    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)
    busqueda.to_csv(DIR_SALIDAS / "busqueda_boosting.csv", index=False)
    importancia.to_csv(DIR_SALIDAS / "importancia_boosting.csv", index=False)

    resultados = {
        "clase_mayoritaria": res_logistica["clase_mayoritaria"],
        BASELINE: res_logistica[BASELINE],
        "gradient_boosting_base": res_boosting,
        "gradient_boosting_con_precio_lc": res_precio,
    }

    try:
        arbol_json_existente = json.loads(
            (DIR_SALIDAS / "metrics_arbol.json").read_text(encoding="utf-8"))
        if "arbol_elegido" in arbol_json_existente["resultados"]:
            resultados["arbol_elegido"] = arbol_json_existente["resultados"]["arbol_elegido"]
    except Exception:
        pass

    ev.guardar_metricas(resultados, DIR_SALIDAS / "metrics_boosting.json", extra={
        "semilla": SEMILLA,
        "tamano_muestra_objetivo": tamano_muestra,
        "cortes": {"validacion": ANIO_CORTE_VALIDACION, "prueba": ANIO_CORTE_PRUEBA},
        "grilla": {k: [str(v) for v in vs] for k, vs in gb.GRILLA.items()},
        "criterio_seleccion": "max ROC-AUC en validación; prueba no se usa",
        "parametros_elegidos": {k: str(v) for k, v in parametros.items()},
        "variables": datos["variables"],
    })

    print("\n" + "=" * 92)
    print("RESULTADOS")
    print("=" * 92)
    print(ev.tabla_metricas(resultados).to_string(index=False))
    print("=" * 92)
    print(f"\nTop 10 importancia por permutación:")
    print(importancia.head(10).round(4).to_string(index=False))
    print(f"\nTiempo total: {time.time() - inicio:.0f}s")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=None, help="ruta al CSV de aceptados")
    parser.add_argument("--muestra", type=int, default=TAMANO_MUESTRA)
    args = parser.parse_args()
    main(args.datos, args.muestra)
