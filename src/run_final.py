"""Entrenamiento, evaluación explícita e inferencia del flujo final congelado."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from . import etapas_finales as ef
from . import analisis_errores as errores
from .config import RAIZ


def carpeta_nueva(destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=False)
    return destino


def destino_unico(base, prefijo):
    return Path(base) / (prefijo + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))


def verificar_configuracion():
    raiz = RAIZ / "outputs"
    final = json.loads((raiz / "evaluacion_final/configuracion_final.json").read_text(encoding="utf-8"))
    optimizacion = json.loads((raiz / "optimizacion_final/configuracion_reproducible.json").read_text(encoding="utf-8"))
    if final["parametros_modelos"] != ef.PARAMETROS or optimizacion["mejores_parametros"] != ef.PARAMETROS:
        raise ValueError("Los parámetros del código difieren de la configuración histórica final.")
    if (final["umbral_faltantes_train_pct"] != 50 or ef.UMBRAL_FALTANTES != 50
            or final["semilla"] != ef.SEMILLA or final["indicadores_faltantes"]
            or final["base_modelo_hasta"] != 2015 or final["calibracion_en"] != 2016
            or final["test_desde"] != 2017
            or any(v != "Sigmoide" for v in final["metodos_calibracion"].values())):
        raise ValueError("Configuración temporal o de preparación incompatible.")
    sig = final["configuracion_calibrador_sigmoide"]
    if any(sig[k] != v for k, v in {"C": 1e6, "solver": "lbfgs", "max_iter": 2000, "random_state": 42}.items()):
        raise ValueError("Configuración sigmoide distinta de la implementación.")
    return final


def guardar_artefacto(pipeline, archivo):
    archivo = Path(archivo)
    with archivo.open("xb") as f:
        joblib.dump(pipeline, f, compress=3)
    return hashlib.sha256(archivo.read_bytes()).hexdigest()


def cargar_artefacto(archivo):
    pipeline = joblib.load(archivo)
    if not isinstance(pipeline, ef.PipelineFinal):
        raise ValueError("El archivo no contiene un PipelineFinal.")
    actual = ef.versiones_dependencias()
    for dependencia in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]:
        if pipeline.metadatos["versiones"][dependencia] != actual[dependencia]:
            raise ValueError(f"Versión incompatible de {dependencia}; use las dependencias del artefacto.")
    return pipeline


def entrenar(datos=None, salida=None):
    configuracion = verificar_configuracion()
    salida = carpeta_nueva(salida or destino_unico(RAIZ / "models", "final"))
    desarrollo, candidatas, fuente = ef.cargar_desarrollo(ruta=datos, ultimo_anio=2016)
    registros = {}
    for pipeline in ef.entrenar_pipelines_finales(desarrollo, candidatas):
        archivo = salida / (pipeline.nombre + ".joblib")
        digest = guardar_artefacto(pipeline, archivo)
        registros[pipeline.nombre] = {"archivo": archivo.name, "sha256": digest,
                                     "metadatos": pipeline.metadatos}
        print("Guardado:", archivo, flush=True)
    manifiesto = {"principal": "gradient_boosting", "modelos": registros,
                  "fuente": Path(fuente).name, "configuracion_historica": configuracion,
                  "test_evaluado_en_esta_ejecucion": False}
    (salida / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Modelos completos:", salida)
    return salida


def cargar_modelos(carpeta, solo_principal=False):
    carpeta = Path(carpeta)
    manifest = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    nombres = [manifest["principal"]] if solo_principal else list(manifest["modelos"])
    modelos = {}
    for nombre in nombres:
        info = manifest["modelos"][nombre]
        archivo = carpeta / info["archivo"]
        if archivo.resolve().parent != carpeta.resolve():
            raise ValueError("Ruta de artefacto fuera de su carpeta.")
        if hashlib.sha256(archivo.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"El artefacto fue modificado: {archivo.name}")
        modelos[nombre] = cargar_artefacto(archivo)
    return modelos


def validar_evaluacion(conjunto, confirmar_test):
    if conjunto not in ["validacion", "test"]:
        raise ValueError("Use conjunto validacion o test.")
    if conjunto == "test" and not confirmar_test:
        raise ValueError("Evaluar test requiere --confirmar-test.")


def _figuras_evaluacion(datos, modelos, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for nombre in modelos:
        y, p = datos["default"], datos[nombre]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        frecuencia, promedio = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[0].plot(fpr, tpr, label=nombre)
        axes[1].plot(recall, precision, label=nombre)
        axes[2].plot(promedio, frecuencia, marker="o", label=nombre)
    axes[0].set(xlabel="Falsos positivos", ylabel="Verdaderos positivos", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precisión", title="Precision–Recall")
    axes[2].set(xlabel="Probabilidad media", ylabel="Frecuencia observada", title="Calibración")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destino / "evaluacion.png", dpi=150)
    plt.close(fig)


def evaluar(modelos, datos, conjunto, confirmar_test=False, salida=None):
    validar_evaluacion(conjunto, confirmar_test)
    if datos is None:
        raise ValueError("La evaluación requiere un CSV explícito con --datos.")
    pipelines = cargar_modelos(modelos)
    salida = carpeta_nueva(salida or destino_unico(RAIZ / "outputs/pipeline_final", conjunto))
    if conjunto == "test":
        conteos, tabla, candidatas, _ = ef.auditar_y_cargar_test(ruta=datos)
        tabla = tabla.loc[tabla.anio.between(2017, 2018)].reset_index(drop=True)
        conteos.to_csv(salida / "auditoria_estados.csv", index=False)
        alcance = "Test histórico 2017–2018; no usado para cambiar el modelo en este ejecutor."
    else:
        tabla, candidatas, _ = ef.cargar_desarrollo(ruta=datos, ultimo_anio=2016)
        tabla = tabla.loc[tabla.anio.eq(2016)].reset_index(drop=True)
        alcance = "2016 es el periodo usado para ajustar el calibrador; evaluación descriptiva, no independiente."
    if tabla.empty or tabla.default.nunique() != 2:
        raise ValueError("El periodo evaluado debe contener ambas clases.")
    X = ef.preparar_X(tabla, candidatas)
    predicciones = tabla[["anio", "default"]].copy()
    filas = []
    for nombre, pipeline in pipelines.items():
        p = pipeline.predecir_preparado(X)
        predicciones[nombre] = p
        filas.append({"modelo": nombre, "metodo": "Sigmoide", **ef.calcular_metricas(tabla.default, p)})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(salida / "metricas.csv", index=False)
    predicciones.to_csv(salida / "predicciones.csv.gz", index=False, compression="gzip")
    segmentos = predicciones.copy()
    if "term" in tabla:
        segmentos["plazo"] = tabla.term.astype("string")
    if "annual_inc" in tabla:
        segmentos["tramo_ingreso"] = pd.cut(tabla.annual_inc.astype(float),
            [-np.inf, 25000, 50000, 75000, 100000, np.inf],
            labels=["≤25 mil", "25–50 mil", "50–75 mil", "75–100 mil", ">100 mil"]).astype("string").fillna("Sin dato")
    if "application_type" in tabla:
        segmentos["tipo_solicitud"] = tabla.application_type.astype("string").fillna("Sin dato")
    columnas = [c for c in ["anio", "plazo", "tramo_ingreso", "tipo_solicitud"] if c in segmentos]
    errores.tabla_metricas_globales(segmentos, modelos=list(pipelines)).to_csv(salida / "clasificacion_0_5.csv", index=False)
    errores.tabla_errores_por_segmento(segmentos, columnas_segmento=columnas,
                                     modelos=list(pipelines)).to_csv(salida / "errores_segmentos.csv", index=False)
    errores.tabla_costos_por_umbral(
        predicciones, modelos=list(pipelines)
    ).to_csv(salida / "costos_por_umbral.csv", index=False)
    _figuras_evaluacion(predicciones, pipelines, salida)
    (salida / "alcance.json").write_text(json.dumps({"conjunto": conjunto, "interpretacion": alcance,
        "umbral_clasificacion": 0.5, "umbral_descriptivo_no_optimizado": True,
        "modelos": str(Path(modelos).resolve()), "versiones": ef.versiones_dependencias()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(alcance)
    print(metricas.to_string(index=False))
    return salida


def predecir(modelos, datos, salida=None, nombre_modelo="gradient_boosting"):
    pipelines = cargar_modelos(modelos)
    if nombre_modelo not in pipelines:
        disponibles = ", ".join(sorted(pipelines))
        raise ValueError(f"Modelo desconocido: {nombre_modelo}. Disponibles: {disponibles}")
    pipeline = pipelines[nombre_modelo]
    columnas = pd.read_csv(datos, nrows=0).columns.tolist()
    faltan = [c for c in pipeline.columnas_entrada if c not in columnas]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")
    tipos = {c: "float64" if c in pipeline.metadatos["numericas_entrada"] and c != "term"
             else "string" for c in pipeline.columnas_entrada}
    salida = Path(salida) if salida else destino_unico(RAIZ / "outputs/predicciones", "prestamos").with_suffix(".csv")
    if salida.exists():
        raise FileExistsError(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("x", encoding="utf-8", newline="") as f:
        offset = 0
        for bloque in pd.read_csv(datos, usecols=pipeline.columnas_entrada, dtype=tipos, chunksize=10000, skipinitialspace=True):
            p = pipeline.predict_proba(bloque)[:, 1]
            pd.DataFrame({"fila_entrada": np.arange(offset, offset + len(bloque)),
                          "probabilidad_charged_off": p}).to_csv(
                              f, index=False, header=offset == 0
                          )
            offset += len(bloque)
    print("Predicciones:", salida)
    return salida


def crear_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=["entrenar", "evaluar", "predecir"], required=True)
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--modelos", type=Path, default=None, help="Carpeta que contiene manifest.json y los joblib")
    parser.add_argument("--salida", type=Path, default=None, help="Carpeta nueva; en predecir, archivo CSV nuevo")
    parser.add_argument("--conjunto", choices=["validacion", "test"], default="validacion")
    parser.add_argument("--confirmar-test", action="store_true")
    parser.add_argument("--modelo-prediccion", choices=list(ef.PARAMETROS),
                        default="gradient_boosting",
                        help="Modelo final a usar en modo predecir.")
    return parser


def main(argv=None):
    parser = crear_parser()
    args = parser.parse_args(argv)
    if args.modo == "entrenar":
        return entrenar(args.datos, args.salida)
    if args.modelos is None or args.datos is None:
        parser.error("evaluar y predecir requieren --modelos y --datos explícitos")
    if args.modo == "evaluar":
        return evaluar(args.modelos, args.datos, args.conjunto, args.confirmar_test, args.salida)
    return predecir(args.modelos, args.datos, args.salida, args.modelo_prediccion)


if __name__ == "__main__":
    main()
