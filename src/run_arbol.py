import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import export_text, plot_tree

from . import arbol as ar
from . import evaluate as ev
from .config import (ANIO_CORTE_PRUEBA, ANIO_CORTE_VALIDACION, DIR_FIGURAS,
                     DIR_SALIDAS, SEMILLA, TAMANO_MUESTRA, localizar_datos)
from .data import dividir_por_anio, escanear_y_muestrear
from .experimentos import entrenar_experimentos, preparar_experimentos

AZUL, ROJO, VERDE, GRIS = "#4C78A8", "#E45756", "#54A24B", "#9AA0A6"
BASELINE = "logistica_sin_indicadores"


def _guardar(fig, nombre):
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / nombre, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {nombre}")


def _texto(valor):
    if valor is None or valor == "None":
        return "None"
    if isinstance(valor, float) and np.isnan(valor):
        return "None"
    return str(int(valor)) if isinstance(valor, (int, float)) else str(valor)


def fig_complejidad(busqueda, seleccion, auc_baseline):
    orden = [str(d) for d in ar.GRILLA["max_depth"] if d is not None] + ["None"]
    etiquetas = [d if d != "None" else "sin\nlímite" for d in orden]
    peso = busqueda["class_weight"].map(_texto)
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
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
    fig.suptitle("10. Complejidad del árbol vs. sobreajuste", fontsize=12)
    fig.tight_layout()
    _guardar(fig, "10_arbol_complejidad.png")


def fig_reglas(arbol, nombres):
    fig, eje = plt.subplots(figsize=(24, 10))
    plot_tree(arbol, max_depth=3, feature_names=list(nombres),
              class_names=["Fully Paid", "Charged Off"], filled=True,
              proportion=True, impurity=False, rounded=True, fontsize=9, ax=eje)
    eje.set_title("11. Primeros 3 niveles del árbol elegido "
                  "(value = proporción [Fully Paid, Charged Off] en el nodo)",
                  fontsize=14)
    _guardar(fig, "11_arbol_reglas.png")


def fig_importancia(importancia, top=15):
    tabla = importancia.head(top).iloc[::-1]
    fig, eje = plt.subplots(figsize=(8, 6.5))
    eje.barh(tabla["variable"], tabla["caida_auc_media"],
             xerr=tabla["caida_auc_desv"], color=AZUL)
    eje.set_xlabel("Caída del ROC-AUC en validación al desordenar la variable")
    eje.set_title("12. Importancia por permutación del árbol elegido", fontsize=11)
    eje.tick_params(labelsize=8)
    _guardar(fig, "12_arbol_importancia.png")


def fig_roc(curvas):
    fig, eje = plt.subplots(figsize=(6.5, 6))
    for (nombre, y, s), color in zip(curvas, [GRIS, AZUL, ROJO]):
        fpr, tpr, _ = roc_curve(y, s)
        eje.plot(fpr, tpr, color=color,
                 label=f"{nombre} (AUC={roc_auc_score(y, s):.4f})")
    eje.plot([0, 1], [0, 1], ":", color="black", linewidth=0.8, label="Azar")
    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos")
    eje.set_title("13. Curvas ROC en validación", fontsize=11)
    eje.legend(fontsize=8, loc="lower right")
    _guardar(fig, "13_comparacion_roc.png")


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

    print("\n[3/6] Preparando datos para el árbol...")
    datos = ar.preparar_datos(particiones, columnas_vacias, incluir_precio_lc=False)
    print(f"  {datos['variables']['predictoras_finales']} predictoras -> "
          f"{datos['variables']['columnas_modelo']} columnas tras one-hot")

    n_combinaciones = int(np.prod([len(v) for v in ar.GRILLA.values()]))
    print(f"\n[4/6] Búsqueda de hiperparámetros ({n_combinaciones} combinaciones, "
          "se elige por ROC-AUC en validación)...")
    busqueda = ar.buscar_hiperparametros(datos)
    mejor = busqueda.iloc[0]
    parametros = ar.parametros_de_fila(mejor)
    print(f"  elegido: {parametros}")
    print(f"  ROC-AUC entrenamiento={mejor['auc_entrenamiento']:.4f}  "
          f"validación={mejor['auc_validacion']:.4f}  "
          f"hojas={int(mejor['hojas'])}")

    print("\n[5/6] Entrenando el árbol elegido y sus referencias...")
    arbol, scores, res_arbol = ar.entrenar_y_evaluar(datos, parametros)
    sin_limite = {"max_depth": None, "min_samples_leaf": 1, "class_weight": None}
    arbol_libre, scores_libre, res_libre = ar.entrenar_y_evaluar(datos, sin_limite)
    datos_precio = ar.preparar_datos(particiones, columnas_vacias,
                                     incluir_precio_lc=True)
    _, _, res_precio = ar.entrenar_y_evaluar(datos_precio, parametros)

    importancia = ar.importancia_por_permutacion(datos, arbol)
    gini = ar.importancia_gini(datos, arbol)

    print("\n[6/6] Figuras y resultados...")
    fig_complejidad(busqueda, parametros, auc_baseline)
    fig_reglas(arbol, datos["nombres"])
    fig_importancia(importancia)
    fig_roc([
        ("Árbol sin restricciones", datos["y"]["validacion"], scores_libre["validacion"]),
        ("Baseline logístico", *scores_logistica[BASELINE]),
        ("Árbol elegido", datos["y"]["validacion"], scores["validacion"]),
    ])

    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)
    busqueda.to_csv(DIR_SALIDAS / "busqueda_arbol.csv", index=False)
    importancia.to_csv(DIR_SALIDAS / "importancia_arbol.csv", index=False)
    gini.head(30).to_csv(DIR_SALIDAS / "importancia_gini_arbol.csv", index=False)
    (DIR_SALIDAS / "reglas_arbol.txt").write_text(
        export_text(arbol, feature_names=list(datos["nombres"]), max_depth=4,
                    show_weights=False, decimals=2), encoding="utf-8")

    resultados = {
        "clase_mayoritaria": res_logistica["clase_mayoritaria"],
        BASELINE: res_logistica[BASELINE],
        "arbol_sin_restricciones": res_libre,
        "arbol_elegido": res_arbol,
        "arbol_elegido_con_precio_lc": res_precio,
    }
    ev.guardar_metricas(resultados, DIR_SALIDAS / "metrics_arbol.json", extra={
        "semilla": SEMILLA,
        "tamano_muestra_objetivo": tamano_muestra,
        "cortes": {"validacion": ANIO_CORTE_VALIDACION, "prueba": ANIO_CORTE_PRUEBA},
        "grilla": {k: [str(v) for v in vs] for k, vs in ar.GRILLA.items()},
        "criterio_seleccion": "max ROC-AUC en validación; prueba no se usa",
        "parametros_elegidos": {k: str(v) for k, v in parametros.items()},
        "arbol_elegido": {"profundidad": int(arbol.get_depth()),
                          "hojas": int(arbol.get_n_leaves())},
        "arbol_sin_restricciones": {"profundidad": int(arbol_libre.get_depth()),
                                    "hojas": int(arbol_libre.get_n_leaves())},
        "variables": datos["variables"],
    })

    print("\n" + "=" * 92)
    print(ev.tabla_metricas(resultados).to_string(index=False))
    print("=" * 92)
    print(f"Árbol elegido: profundidad {arbol.get_depth()}, {arbol.get_n_leaves()} hojas")
    print(f"Árbol sin restricciones: profundidad {arbol_libre.get_depth()}, "
          f"{arbol_libre.get_n_leaves()} hojas")
    print("\nTop 10 importancia por permutación:")
    print(importancia.head(10).round(4).to_string(index=False))
    print(f"\nTiempo total: {time.time() - inicio:.0f}s")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=None, help="ruta al CSV de aceptados")
    parser.add_argument("--muestra", type=int, default=TAMANO_MUESTRA)
    args = parser.parse_args()
    main(args.datos, args.muestra)
