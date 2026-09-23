"""Pipeline reproducible de extremo a extremo para la entrega previa.

Ejecutar desde la raiz del proyecto:

    python -m src.run_baseline

Produce:
  outputs/metrics.json     metricas de todos los modelos y conjuntos
  outputs/auditoria.json   cifras del dataset citadas en proposal.md
  reports/figures/*.png    visualizaciones del analisis exploratorio
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as ft
from .config import (ANIO_CORTE_PRUEBA, ANIO_CORTE_VALIDACION, DIR_FIGURAS,
                     DIR_SALIDAS, SEMILLA, TAMANO_MUESTRA, UMBRAL_FALTANTES,
                     UMBRAL_REVISION, localizar_datos)
from .data import dividir_por_anio, escanear_y_muestrear
from .experimentos import preparar_experimentos, entrenar_experimentos

AZUL, ROJO, GRIS = "#4C78A8", "#E45756", "#9AA0A6"


def _guardar(fig, nombre):
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {ruta.name}")



# Visualizaciones
def fig_distribucion_objetivo(muestra):
    conteo = muestra["default"].value_counts().reindex([0, 1], fill_value=0)
    fig, eje = plt.subplots(figsize=(6, 4))
    barras = eje.bar(["Fully Paid (0)", "Charged Off (1)"], conteo.values,
                     color=[AZUL, ROJO])
    total = conteo.sum()
    for barra, valor in zip(barras, conteo.values):
        eje.text(barra.get_x() + barra.get_width() / 2, valor,
                 f"{valor:,}\n({valor / total:.1%})", ha="center", va="bottom")
    eje.set_title("1. Distribucion del objetivo en la muestra")
    eje.set_ylabel("Prestamos")
    eje.margins(y=0.18)
    _guardar(fig, "01_distribucion_objetivo.png")


def fig_default_por_anio(muestra):
    agrupado = muestra.groupby("anio").agg(
        tasa=("default", "mean"), n=("default", "size"))
    fig, eje = plt.subplots(figsize=(9, 4.5))
    eje.bar(agrupado.index, agrupado["n"], color=GRIS, alpha=0.55,
            label="Prestamos con desenlace")
    eje.set_ylabel("Prestamos en la muestra")
    eje.set_xlabel("Anio de otorgamiento")
    eje2 = eje.twinx()
    eje2.plot(agrupado.index, agrupado["tasa"] * 100, marker="o", color=ROJO,
              label="Tasa de default (%)")
    eje2.set_ylabel("Tasa de default (%)", color=ROJO)
    for corte, texto in [(ANIO_CORTE_VALIDACION, "validacion"),
                         (ANIO_CORTE_PRUEBA, "prueba")]:
        eje.axvline(corte - 0.5, color="black", linestyle="--", linewidth=1)
        eje.text(corte - 0.4, eje.get_ylim()[1] * 0.95, texto, fontsize=8,
                 rotation=90, va="top")
    eje.set_title("2. Volumen y tasa de default por anio (con cortes temporales)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
    _guardar(fig, "02_default_por_anio.png")


def fig_horizonte_observacion(muestra):
    """Evidencia del sesgo de supervivencia en los anios recientes."""
    tabla = pd.crosstab(muestra["anio"], muestra["term"], normalize="index") * 100
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    tabla.plot(kind="bar", stacked=True, ax=ejes[0], color=[AZUL, ROJO], width=0.85)
    ejes[0].set_title("Composicion por plazo (%)")
    ejes[0].set_ylabel("% de prestamos con desenlace")
    ejes[0].set_xlabel("Anio de otorgamiento")
    ejes[0].legend(title="Plazo (meses)", fontsize=8)

    por_plazo = muestra.groupby(["anio", "term"])["default"].mean().unstack() * 100
    por_plazo.plot(ax=ejes[1], marker="o", color=[AZUL, ROJO])
    ejes[1].set_title("Tasa de default por plazo y anio (%)")
    ejes[1].set_ylabel("Tasa de default (%)")
    ejes[1].set_xlabel("Anio de otorgamiento")
    ejes[1].legend(title="Plazo (meses)", fontsize=8)
    fig.suptitle("3. Plazos y resultados por año: revisar seguimiento disponible", fontsize=11)
    _guardar(fig, "03_horizonte_observacion.png")


def fig_faltantes(muestra):
    porcentaje = (muestra.isna().mean() * 100).sort_values(ascending=False).head(30)
    fig, eje = plt.subplots(figsize=(8, 8))
    eje.barh(porcentaje.index[::-1], porcentaje.values[::-1], color=AZUL)
    eje.axvline(UMBRAL_FALTANTES, color=ROJO, linestyle="--",
                label=f"{UMBRAL_FALTANTES}%: excluir")
    eje.axvline(UMBRAL_REVISION, color="orange", linestyle="--",
                label=f"{UMBRAL_REVISION}%: imputar + indicador")
    eje.set_xlabel("% de valores faltantes en la muestra")
    eje.set_title("4. Las 30 variables con mas faltantes")
    eje.legend(fontsize=8)
    _guardar(fig, "04_faltantes.png")


def fig_variables_clave(entrenamiento):
    variables = [("fico_range_low", "FICO (limite inferior)"),
                 ("dti", "Ratio deuda / ingreso"),
                 ("annual_inc", "Ingreso anual (log10)"),
                 ("revol_util", "Utilizacion revolvente (%)")]
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7))
    for eje, (columna, titulo) in zip(ejes.ravel(), variables):
        datos = entrenamiento[[columna, "default"]].dropna().copy()
        if columna == "annual_inc":
            datos = datos[datos[columna] > 0]
            datos[columna] = np.log10(datos[columna])
        bajo, alto = datos[columna].quantile([0.01, 0.99])
        datos = datos[datos[columna].between(bajo, alto)]
        for valor, color, etiqueta in [(0, AZUL, "Fully Paid"),
                                       (1, ROJO, "Charged Off")]:
            eje.hist(datos.loc[datos["default"] == valor, columna], bins=50,
                     density=True, alpha=0.55, color=color, label=etiqueta)
        eje.set_title(titulo)
        eje.legend(fontsize=8)
    fig.suptitle("5. Distribucion de variables clave por clase (entrenamiento)",
                 fontsize=12)
    fig.tight_layout()
    _guardar(fig, "05_variables_clave.png")


def fig_categoricas(entrenamiento):
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    for eje, columna, titulo in [
            (ejes[0], "grade", "Calificacion asignada por LC"),
            (ejes[1], "purpose", "Proposito del prestamo"),
            (ejes[2], "home_ownership", "Tenencia de vivienda")]:
        tabla = entrenamiento.groupby(columna).agg(
            tasa=("default", "mean"), n=("default", "size"))
        tabla = tabla[tabla["n"] >= 50].sort_values("tasa")
        eje.barh(tabla.index.astype(str), tabla["tasa"] * 100, color=AZUL)
        eje.axvline(entrenamiento["default"].mean() * 100, color=ROJO,
                    linestyle="--", linewidth=1, label="Tasa global")
        eje.set_title(titulo, fontsize=10)
        eje.set_xlabel("Tasa de default (%)")
        eje.tick_params(labelsize=8)
        eje.legend(fontsize=7)
    fig.suptitle("6. Tasa de default por categoria (entrenamiento)", fontsize=12)
    fig.tight_layout()
    _guardar(fig, "06_categoricas.png")


def fig_outliers(entrenamiento):
    variables = ["loan_amnt", "annual_inc", "dti", "revol_bal",
                 "fico_range_low", "revol_util"]
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
    for eje, columna in zip(ejes.ravel(), variables):
        datos = entrenamiento[columna].dropna()
        eje.boxplot(datos, vert=True, showfliers=True,
                    flierprops={"markersize": 2, "alpha": 0.3})
        q1, q3 = datos.quantile([0.25, 0.75])
        iqr = q3 - q1
        fuera = ((datos < q1 - 1.5 * iqr) | (datos > q3 + 1.5 * iqr)).mean() * 100
        eje.set_title(f"{columna}\n{fuera:.1f}% fuera del rango IQR", fontsize=9)
    fig.suptitle("7. Valores extremos por regla IQR (no se eliminan filas)",
                 fontsize=12)
    fig.tight_layout()
    _guardar(fig, "07_outliers.png")


def fig_curvas(scores):
    """Curvas ROC y Precision-Recall en validacion.

    La referencia sin capacidad predictiva se dibuja como recta: la diagonal
    en ROC y una horizontal en la prevalencia de la clase positiva en PR. Un
    score constante produce una curva PR degenerada que enganaria la lectura,
    por eso `clase_mayoritaria` no se traza como curva.
    """
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
    prevalencia = None
    for nombre, (y, s) in scores.items():
        if nombre == "clase_mayoritaria":
            prevalencia = float(np.mean(y))
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ejes[0].plot(fpr, tpr, label=nombre)
        precision, recall, _ = precision_recall_curve(y, s)
        ejes[1].plot(recall, precision, label=nombre)
    if prevalencia is not None:
        ejes[1].axhline(prevalencia, linestyle="--", color=GRIS,
                        label=f"Sin capacidad predictiva ({prevalencia:.1%})")
        ejes[1].set_ylim(0, 1)
    ejes[0].plot([0, 1], [0, 1], "--", color=GRIS, label="Azar")
    ejes[0].set_xlabel("Tasa de falsos positivos")
    ejes[0].set_ylabel("Tasa de verdaderos positivos")
    ejes[0].set_title("Curva ROC (validacion)")
    ejes[0].legend(fontsize=8)
    ejes[1].set_xlabel("Recall")
    ejes[1].set_ylabel("Precision")
    ejes[1].set_title("Curva Precision-Recall (validacion)")
    ejes[1].legend(fontsize=8)
    fig.suptitle("8. Desempeno del baseline en validacion", fontsize=12)
    fig.tight_layout()
    _guardar(fig, "08_curvas_baseline.png")


def fig_coeficientes(tabla):
    fig, eje = plt.subplots(figsize=(8, 7))
    tabla = tabla.sort_values("coeficiente")
    colores = [ROJO if c > 0 else AZUL for c in tabla["coeficiente"]]
    eje.barh(tabla["variable"], tabla["coeficiente"], color=colores)
    eje.axvline(0, color="black", linewidth=0.8)
    eje.set_xlabel("Coeficiente: numéricas escaladas y categorías frente a referencia")
    eje.set_title("9. Variables mas influyentes del baseline\n"
                  "rojo = aumenta el riesgo, azul = lo reduce", fontsize=11)
    eje.tick_params(labelsize=8)
    _guardar(fig, "09_coeficientes_baseline.png")



# Pipeline principal
def main(ruta_datos=None, tamano_muestra=TAMANO_MUESTRA):
    inicio = time.time()
    ruta = localizar_datos(ruta_datos)
    print(f"Dataset: {ruta}")
    print(f"Tamano en disco: {ruta.stat().st_size / 1e6:.0f} MB\n")

    print("[1/6] Escaneando el archivo completo y construyendo la muestra...")
    muestra, auditoria = escanear_y_muestrear(ruta, tamano_muestra=tamano_muestra)
    print(f"  {auditoria['total_filas']:,} filas x "
          f"{auditoria['total_columnas']} columnas en el archivo")
    print(f"  {auditoria['total_elegibles']:,} prestamos con desenlace final")
    print(f"  muestra: {len(muestra):,} filas "
          f"({auditoria['fraccion_muestreo']:.2%} de los elegibles)")
    print(f"  tasa de default en la muestra: {muestra['default'].mean():.4f}\n")

    print("[2/6] Particion temporal por anio de otorgamiento...")
    entrenamiento, validacion, prueba = dividir_por_anio(
        muestra, ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA)
    for nombre, parte in [("entrenamiento", entrenamiento),
                          ("validacion", validacion), ("prueba", prueba)]:
        print(f"  {nombre:<14} {len(parte):>8,} filas  "
              f"default={parte['default'].mean():.4f}  "
              f"anios {int(parte['anio'].min())}-{int(parte['anio'].max())}")
    print()

    print("[3/6] Generando visualizaciones...")
    fig_distribucion_objetivo(muestra)
    fig_default_por_anio(muestra)
    fig_horizonte_observacion(muestra)
    fig_faltantes(muestra)
    fig_variables_clave(entrenamiento)
    fig_categoricas(entrenamiento)
    columnas_outliers = ['loan_amnt', 'annual_inc', 'dti', 'revol_bal', 'fico_range_low', 'revol_util']
    datos_outliers = entrenamiento[columnas_outliers].copy()
    datos_outliers = datos_outliers.fillna(datos_outliers.median())
    fig_outliers(datos_outliers)
    print()

    print("[4/6] Seleccionando predictoras y controlando leakage...")
    faltantes_globales = auditoria["faltantes_globales"]
    columnas_vacias = faltantes_globales[
        faltantes_globales == auditoria["total_filas"]].index.tolist()
    exclusiones = ft.tabla_exclusiones(muestra.columns.tolist(), columnas_vacias)
    print(exclusiones["grupo"].value_counts().to_string())

    particiones = {'entrenamiento': entrenamiento, 'validacion': validacion, 'prueba': prueba}
    preparados = preparar_experimentos(particiones, columnas_vacias)
    print('[5/6] Entrenando los tres escenarios y evaluando la referencia...')
    resultados, modelos, scores_validacion = entrenar_experimentos(preparados)
    resumen_variables = {nombre: datos['variables'] for nombre, datos in preparados.items()}
    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)
    base = 'logistica_sin_indicadores'
    tabla_coef = pd.DataFrame({
        'variable': preparados[base]['X']['entrenamiento'].columns,
        'coeficiente': modelos[base].coef_[0],
    })
    tabla_coef['odds_ratio'] = np.exp(tabla_coef['coeficiente'])
    tabla_coef['magnitud'] = tabla_coef['coeficiente'].abs()
    tabla_coef = tabla_coef.sort_values('magnitud', ascending=False).head(20)
    fig_coeficientes(tabla_coef)
    tabla_coef.to_csv(DIR_SALIDAS / 'coeficientes_baseline.csv', index=False)
    for nombre, datos in preparados.items():
        datos['faltantes'].to_csv(DIR_SALIDAS / f'decisiones_faltantes_{nombre}.csv')
    preparados[base]['faltantes'].to_csv(DIR_SALIDAS / 'decisiones_faltantes.csv')

    fig_curvas(scores_validacion)

    print("\n[6/6] Guardando resultados...")
    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)
    exclusiones.to_csv(DIR_SALIDAS / "exclusiones_variables.csv",
                       index=False, encoding="utf-8")
    ev.guardar_metricas(resultados, DIR_SALIDAS / "metrics.json", extra={
        "semilla": SEMILLA,
        "tamano_muestra_objetivo": tamano_muestra,
        "cortes": {"validacion": ANIO_CORTE_VALIDACION,
                   "prueba": ANIO_CORTE_PRUEBA},
        "variables": resumen_variables,
    })

    auditoria_json = {
        "archivo": ruta.name,
        "total_filas": auditoria["total_filas"],
        "total_columnas": auditoria["total_columnas"],
        "filas_resumen": auditoria["filas_resumen"],
        "total_elegibles": auditoria["total_elegibles"],
        "fraccion_muestreo": auditoria["fraccion_muestreo"],
        "filas_muestra": int(len(muestra)),
        "tasa_default_muestra": float(muestra["default"].mean()),
        "estados": {str(k): int(v) for k, v in auditoria["estados"].items()},
        "alertas_calidad": {str(k): int(v) for k, v in auditoria["alertas"].items()},
        "filas_con_alerta": auditoria["filas_con_alerta"],
        "valores_policy_code": [str(v) for v in auditoria["valores_policy"]],
        "columnas_vacias": columnas_vacias,
        "particion": {
            nombre: {"filas": int(len(parte)),
                     "tasa_default": float(parte["default"].mean()),
                     "anio_min": int(parte["anio"].min()),
                     "anio_max": int(parte["anio"].max())}
            for nombre, parte in [("entrenamiento", entrenamiento),
                                  ("validacion", validacion), ("prueba", prueba)]},
        "default_por_anio": {
            str(int(a)): {"n": int(g.size), "tasa": float(g.mean())}
            for a, g in muestra.groupby("anio")["default"]},
    }
    (DIR_SALIDAS / "auditoria.json").write_text(
        json.dumps(auditoria_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print("RESULTADOS")
    print("=" * 78)
    print(ev.tabla_metricas(resultados).to_string(index=False))
    print("=" * 78)
    print(f"\nTiempo total: {time.time() - inicio:.0f}s")
    return resultados, auditoria_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=None, help="ruta al CSV de aceptados")
    parser.add_argument("--muestra", type=int, default=TAMANO_MUESTRA,
                        help="tamano objetivo de la muestra")
    args = parser.parse_args()
    main(args.datos, args.muestra)
