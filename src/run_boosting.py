"""Comparación manual con objetivo 220000. No se ejecuta al importar.

python -m src.run_boosting --muestra 220000
"""
import argparse
import json
import time
import gc
from pathlib import Path

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


def _guardar(fig, nombre, carpeta=None):
    carpeta = Path(carpeta) if carpeta else DIR_FIGURAS / "gradient_boosting_220000"
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / nombre, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {nombre}")


def fig_busqueda(busqueda, auc_baseline, carpeta=None):
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
    _guardar(fig, "14_boosting_busqueda.png", carpeta)


def fig_importancia(importancia, top=15, carpeta=None):
    """Top variables por importancia de permutacion."""
    tabla = importancia.head(top).iloc[::-1]
    fig, eje = plt.subplots(figsize=(8, 6.5))
    eje.barh(tabla["variable"], tabla["caida_auc_media"],
             xerr=tabla["caida_auc_desv"], color=VERDE)
    eje.set_xlabel("Caída del ROC-AUC en validación al desordenar la variable")
    eje.set_title("15. Importancia por permutación — Gradient Boosting elegido",
                  fontsize=11)
    eje.tick_params(labelsize=8)
    _guardar(fig, "15_boosting_importancia.png", carpeta)


def fig_roc_comparacion(curvas, carpeta=None):
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
    _guardar(fig, "16_boosting_comparacion_roc.png", carpeta)


def crear_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=None, help="CSV fuente del baseline y árbol")
    parser.add_argument("--muestra", type=int, choices=[220000], default=220000)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--figuras", type=Path, default=None)
    parser.add_argument("--modo", choices=["completo", "prueba"], default="completo",
                        help="completo: 108 combinaciones; prueba: una, mismas observaciones")
    return parser


def rutas_resultados(tamano_muestra=220000, modo="completo", salida=None, figuras=None):
    sufijo = f"gradient_boosting_{tamano_muestra}" + ("_prueba" if modo == "prueba" else "")
    return (Path(salida) if salida else DIR_SALIDAS / sufijo,
            Path(figuras) if figuras else DIR_FIGURAS / sufijo)


def _validar_destinos(salida, figuras):
    destinos = [Path(salida).resolve(), Path(figuras).resolve()]
    for destino in destinos:
        if destino in [DIR_SALIDAS.resolve(), DIR_FIGURAS.resolve()]:
            raise ValueError("Debe usar subcarpetas independientes para proteger resultados anteriores.")
        if destino.exists() and (not destino.is_dir() or any(destino.iterdir())):
            raise FileExistsError(f"Destino ocupado: {destino}. Use otra carpeta.")
    if destinos[0] == destinos[1] or destinos[0] in destinos[1].parents or destinos[1] in destinos[0].parents:
        raise ValueError("Las carpetas de métricas y figuras deben ser independientes.")


def main(ruta_datos=None, tamano_muestra=220000, salida=None, figuras=None, modo="completo"):
    from . import arbol as ar
    from .comparabilidad_muestra import verificar_muestra
    if tamano_muestra != 220000 or modo not in ["completo", "prueba"]:
        raise ValueError("Use muestra 220000 y modo completo o prueba.")
    salida, figuras = rutas_resultados(tamano_muestra, modo, salida, figuras)
    _validar_destinos(salida, figuras)
    ruta = localizar_datos(ruta_datos)
    referencias = [json.loads((DIR_SALIDAS / nombre).read_text(encoding="utf-8"))
                   for nombre in ["metrics.json", "metrics_arbol.json"]]
    inicio = time.perf_counter()
    muestra, auditoria = escanear_y_muestrear(ruta, tamano_muestra=tamano_muestra, verbose=True)
    manifiesto = verificar_muestra(ruta, muestra, auditoria, tamano_muestra, referencias)
    partes = dividir_por_anio(muestra, ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA)
    particiones = dict(zip(["entrenamiento", "validacion", "prueba"], partes))
    faltantes = auditoria["faltantes_globales"]
    vacias = faltantes[faltantes == auditoria["total_filas"]].index.tolist()
    filas_muestra = len(muestra)
    del muestra, auditoria, partes
    gc.collect()
    preparados = preparar_experimentos(particiones, vacias, escenarios=[(BASELINE, False, False)])
    columnas_logistica = preparados[BASELINE]["preprocesador"].feature_names_in_.tolist()
    resultados, modelos_logistica, scores_logistica = entrenar_experimentos(preparados)
    curvas = [("Baseline logístico", *scores_logistica[BASELINE])]
    del preparados, modelos_logistica, scores_logistica
    gc.collect()
    datos_arbol = ar.preparar_datos(particiones, vacias, incluir_precio_lc=False)
    params_arbol = referencias[1]["parametros_elegidos"]
    params_arbol = {k: None if v in [None, "None"] else v if k == "class_weight" else int(v)
                   for k, v in params_arbol.items()}
    modelo_arbol, scores_arbol, res_arbol = ar.entrenar_y_evaluar(datos_arbol, params_arbol)
    resultados["arbol_elegido"] = res_arbol
    curvas.append(("Árbol elegido (parámetros fijados)", datos_arbol["y"]["validacion"], scores_arbol["validacion"]))
    columnas_arbol = datos_arbol["columnas"]
    del datos_arbol, modelo_arbol, scores_arbol
    gc.collect()
    datos = gb.preparar_datos(particiones, vacias, incluir_precio_lc=False)
    if (datos["columnas"] != columnas_arbol or datos["columnas"] != columnas_logistica
            or datos["variables"]["predictoras_finales"] != 65):
        raise ValueError("El conjunto de variables no coincide con el experimento inicial.")
    grilla = gb.GRILLA if modo == "completo" else {k: [v[0]] for k, v in gb.GRILLA.items()}
    print(f"Muestra verificada: {filas_muestra:,}; búsqueda {modo}: {np.prod([len(v) for v in grilla.values()])} configuraciones")
    print("Memoria de matrices GB (MiB):", {k: round(x.nbytes / 1024**2, 1) for k, x in datos["X"].items()})
    busqueda = gb.buscar_hiperparametros(datos, grilla=grilla)
    parametros = gb.parametros_de_fila(busqueda.iloc[0])
    modelo, scores, metricas = gb.entrenar_y_evaluar(datos, parametros)
    resultados["gradient_boosting_base"] = metricas
    curvas.append(("Gradient Boosting elegido", datos["y"]["validacion"], scores["validacion"]))
    importancia = gb.importancia_por_permutacion(datos, modelo)
    salida.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    busqueda.to_csv(salida / "busqueda_boosting.csv", index=False)
    importancia.to_csv(salida / "importancia_boosting.csv", index=False)
    (salida / "muestra_verificada.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    ev.guardar_metricas(resultados, salida / "metrics_boosting.json", extra={
        "semilla": SEMILLA, "tamano_muestra_objetivo": tamano_muestra,
        "filas_muestra": filas_muestra, "modo": modo,
        "cortes": {"validacion": ANIO_CORTE_VALIDACION, "prueba": ANIO_CORTE_PRUEBA},
        "grilla": grilla, "parametros_elegidos": parametros,
        "parametros_arbol_fijados": params_arbol, "variables": datos["variables"],
        "criterio_seleccion": "máximo ROC-AUC de validación; test solo después de fijar parámetros",
        "segundos_totales": time.perf_counter() - inicio})
    fig_busqueda(busqueda, resultados[BASELINE]["validacion"]["roc_auc"], carpeta=figuras)
    fig_importancia(importancia, carpeta=figuras)
    fig_roc_comparacion(curvas, carpeta=figuras)
    print(ev.tabla_metricas(resultados).to_string(index=False))
    print("Métricas:", salida, "Figuras:", figuras)
    return resultados


if __name__ == "__main__":
    args = crear_parser().parse_args()
    main(args.datos, args.muestra, args.salida, args.figuras, args.modo)
