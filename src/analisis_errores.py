"""Analisis de errores por segmentos para las predicciones finales guardadas.

El analisis usa un umbral fijo de 0.5, sin ajustarlo con el conjunto de prueba.
Solo carga del CSV las columnas de segmentacion y el estado del prestamo.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .config import MAPA_OBJETIVO, localizar_datos

MODELOS = ["regresion_logistica", "svm_lineal", "gradient_boosting"]
UMBRAL_FIJO = 0.5


def cargar_segmentos_test(predicciones, ruta=None, chunksize=100_000):
    """Carga term, ingreso y tipo de solicitud, validando el orden de filas.

    El flujo final concatena los préstamos resueltos en el orden del CSV. Se
    reconstruye ese mismo orden y se compara año y etiqueta antes de asociar
    metadatos a las probabilidades; así no se cruzan filas silenciosamente.
    """
    ruta = localizar_datos(ruta) if ruta is not None else localizar_datos()
    columnas = ["issue_d", "loan_status", "term", "annual_inc", "application_type"]
    tipos = {c: "string" for c in columnas if c != "annual_inc"}
    tipos["annual_inc"] = "float64"
    partes = []
    lector = pd.read_csv(ruta, usecols=columnas, dtype=tipos,
                         chunksize=chunksize, skipinitialspace=True,
                         low_memory=False)
    for bloque in lector:
        fecha = pd.to_datetime(bloque["issue_d"], format="%b-%Y", errors="coerce")
        mascara = fecha.dt.year.ge(2017) & bloque["loan_status"].isin(MAPA_OBJETIVO)
        if not mascara.any():
            continue
        parte = bloque.loc[mascara, ["loan_status", "term", "annual_inc",
                                     "application_type"]].copy()
        parte["anio"] = fecha.loc[mascara].dt.year.astype("int16")
        parte["default"] = parte["loan_status"].map(MAPA_OBJETIVO).astype("int8")
        partes.append(parte.drop(columns="loan_status"))
    metadatos = pd.concat(partes, ignore_index=True)
    pred = predicciones.reset_index(drop=True).copy()
    if len(metadatos) != len(pred):
        raise ValueError(f"No coincide el numero de filas: metadatos={len(metadatos):,}, "
                         f"predicciones={len(pred):,}.")
    if not np.array_equal(metadatos["anio"].to_numpy(), pred["anio"].to_numpy()):
        raise ValueError("El orden de anios no coincide con el archivo de predicciones.")
    if not np.array_equal(metadatos["default"].to_numpy(), pred["default"].to_numpy()):
        raise ValueError("Las etiquetas del CSV no coinciden con las predicciones guardadas.")

    salida = pd.concat([pred, metadatos.drop(columns=["anio", "default"])], axis=1)
    salida["plazo"] = (salida["term"].astype("string").str.extract(r"(36|60)", expand=False)
                       .fillna("No identificado"))
    ingreso = pd.to_numeric(salida["annual_inc"], errors="coerce")
    salida["tramo_ingreso"] = pd.cut(
        ingreso, bins=[-np.inf, 25_000, 50_000, 75_000, 100_000, np.inf],
        labels=["≤25 mil", "25–50 mil", "50–75 mil", "75–100 mil", ">100 mil"]
    ).astype("string").fillna("Sin dato")
    salida["tipo_solicitud"] = (salida["application_type"].astype("string")
                                 .fillna("Sin dato"))
    return salida


def _metricas_grupo(y, p, umbral=UMBRAL_FIJO):
    y = np.asarray(y, dtype="int8")
    p = np.asarray(p, dtype="float64")
    pred = (p >= umbral).astype("int8")
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "incumplimiento_pct": float(y.mean() * 100) if len(y) else np.nan,
        "ROC_AUC": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "PR_AUC": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "precision_umbral_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_umbral_0_5": float(recall_score(y, pred, zero_division=0)),
        "F1_umbral_0_5": float(f1_score(y, pred, zero_division=0)),
        "verdaderos_negativos": int(vn), "falsos_positivos": int(fp),
        "falsos_negativos": int(fn), "verdaderos_positivos": int(vp),
    }


def tabla_errores_por_segmento(datos, columnas_segmento=("anio", "plazo", "tramo_ingreso",
                                                          "tipo_solicitud"),
                               modelos=MODELOS, umbral=UMBRAL_FIJO):
    """Devuelve conteos y métricas en cada grupo, para cada modelo."""
    registros = []
    for modelo in modelos:
        if modelo not in datos:
            continue
        for variable in columnas_segmento:
            for grupo, bloque in datos.groupby(variable, dropna=False, observed=True):
                registro = {"modelo": modelo, "segmento": variable, "grupo": str(grupo)}
                registro.update(_metricas_grupo(bloque["default"], bloque[modelo], umbral))
                registros.append(registro)
    return pd.DataFrame(registros)


def tabla_metricas_globales(datos, modelos=MODELOS, umbral=UMBRAL_FIJO):
    filas = []
    for modelo in modelos:
        if modelo in datos:
            filas.append({"modelo": modelo, "umbral_fijo": umbral,
                          **_metricas_grupo(datos["default"], datos[modelo], umbral)})
    return pd.DataFrame(filas)


def ejecutar_analisis(ruta=None, archivo_predicciones=None, salida="outputs/analisis_errores",
                      chunksize=100_000):
    """Lee predicciones existentes, verifica filas y exporta tablas agregadas."""
    ruta = localizar_datos(ruta) if ruta is not None else localizar_datos()
    if archivo_predicciones is None:
        archivo_predicciones = Path("outputs/evaluacion_final/predicciones_test.csv.gz")
    pred = pd.read_csv(archivo_predicciones)
    datos = cargar_segmentos_test(pred, ruta=ruta, chunksize=chunksize)
    metricas = tabla_metricas_globales(datos)
    segmentos = tabla_errores_por_segmento(datos)
    destino = Path(salida)
    destino.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(destino / "metricas_globales.csv", index=False)
    segmentos.to_csv(destino / "metricas_por_segmento.csv", index=False)
    return datos, metricas, segmentos


def main():
    parser = argparse.ArgumentParser(description="Analiza errores por segmento en el test guardado.")
    parser.add_argument("--datos", default=None, help="Ruta del CSV Lending Club")
    parser.add_argument("--predicciones", default=None,
                        help="CSV de probabilidades guardadas del test")
    parser.add_argument("--salida", default="outputs/analisis_errores")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    datos, globales, segmentos = ejecutar_analisis(
        ruta=args.datos, archivo_predicciones=args.predicciones,
        salida=args.salida, chunksize=args.chunksize
    )
    print(f"Filas verificadas: {len(datos):,}")
    print(globales.to_string(index=False))
    print(f"Resultados agregados: {Path(args.salida).resolve()}")


if __name__ == "__main__":
    main()
