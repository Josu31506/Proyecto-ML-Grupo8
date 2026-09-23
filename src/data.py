"""Carga del dataset de Lending Club.

La primera pasada por el archivo recolecta: auditoria global
(dimensiones, tipos, faltantes, estados), alertas de calidad. La segunda toma una muestra aleatoria proporcional por bloques. El CSV original nunca se modifica.
"""
import numpy as np
import pandas as pd

from .config import (ESTADOS_CONSERVADOS, MAPA_OBJETIVO, SEMILLA,
                     TAMANO_BLOQUE, TAMANO_MUESTRA)

COLUMNAS_FECHA = ["issue_d", "earliest_cr_line", "sec_app_earliest_cr_line"]

# Se fuerzan a texto: son codigos o identificadores, no cantidades.
FORZAR_TEXTO = ["id", "member_id", "policy_code", "zip_code", "term"] + COLUMNAS_FECHA

# Reglas de coherencia aplicadas a los prestamos elegibles.
COLUMNAS_REVISION = ["loan_status", "loan_amnt", "annual_inc", "dti", "revol_bal",
                     "fico_range_low", "fico_range_high", "term",
                     "earliest_cr_line", "issue_d"]


def inferir_tipos(ruta, filas=100_000):
    """Deduce el dtype de lectura de cada columna con una muestra inicial."""
    vista = pd.read_csv(ruta, nrows=filas, low_memory=False, skipinitialspace=True)
    tipos = {}
    for columna in vista.columns:
        es_numerica = pd.api.types.is_numeric_dtype(vista[columna])
        tipos[columna] = "float64" if es_numerica else "string"
    for columna in FORZAR_TEXTO:
        if columna in tipos:
            tipos[columna] = "string"
    return tipos, vista.columns.tolist()


def _normalizar(bloque):
    """Convierte fechas y plazo a su tipo adecuado dentro de un bloque."""
    for columna in COLUMNAS_FECHA:
        if columna in bloque.columns:
            bloque[columna] = pd.to_datetime(bloque[columna], format="%b-%Y",
                                             errors="coerce")
    if "term" in bloque.columns:
        bloque["term"] = pd.to_numeric(
            bloque["term"].str.strip().str.replace(" months", "", regex=False),
            errors="coerce").astype("float64")
    return bloque


def _alertas_calidad(bloque):
    """Reglas fijas de coherencia. Marcan filas para revisar, no para borrar."""
    return pd.DataFrame({
        "ingreso_negativo": bloque["annual_inc"] < 0,
        "ingreso_cero": bloque["annual_inc"] == 0,
        "monto_no_positivo": bloque["loan_amnt"] <= 0,
        "dti_negativo": bloque["dti"] < 0,
        "saldo_revolvente_negativo": bloque["revol_bal"] < 0,
        "fico_invertido": bloque["fico_range_low"] > bloque["fico_range_high"],
        "fico_fuera_de_rango": ((bloque["fico_range_low"] < 300)
                                | (bloque["fico_range_low"] > 850)
                                | (bloque["fico_range_high"] < 300)
                                | (bloque["fico_range_high"] > 850)),
        "fecha_credito_posterior": bloque["earliest_cr_line"] > bloque["issue_d"],
        "plazo_no_esperado": bloque["term"].notna() & ~bloque["term"].isin([36, 60]),
    }).fillna(False)


def escanear_y_muestrear(ruta, tamano_muestra=TAMANO_MUESTRA,
                         tamano_bloque=TAMANO_BLOQUE, semilla=SEMILLA, verbose=True):
    """Recorre el archivo completo en dos pasadas y devuelve auditoria + muestra.

    El muestreo es proporcional por bloque: de los prestamos elegibles de cada
    bloque se toma la misma fraccion, sin reemplazo. Los bloques solo limitan
    el uso de memoria; no cambian el diseno del muestreo.
    """
    tipos, columnas = inferir_tipos(ruta)

    total_filas = 0
    filas_resumen = 0
    faltantes = pd.Series(dtype="float64")
    estados = pd.Series(dtype="float64")
    alertas = pd.Series(dtype="float64")
    filas_con_alerta = 0
    valores_policy = set()
    conteos_anio_estado = []


    lector = pd.read_csv(ruta, dtype=tipos, chunksize=tamano_bloque,
                         skipinitialspace=True)
    for numero, bloque in enumerate(lector):
        total_filas += len(bloque)
        faltantes = faltantes.add(bloque.isna().sum(), fill_value=0)
        estados = estados.add(
            bloque["loan_status"].fillna("SIN ESTADO").value_counts(), fill_value=0)
        valores_policy.update(bloque["policy_code"].dropna().unique())
        filas_resumen += int(
            bloque[["issue_d", "loan_status", "loan_amnt"]].isna().all(axis=1).sum())

        bloque = _normalizar(bloque)
        elegibles = bloque[bloque["loan_status"].isin(ESTADOS_CONSERVADOS)]
        if len(elegibles):
            reglas = _alertas_calidad(elegibles)
            alertas = alertas.add(reglas.sum(), fill_value=0)
            filas_con_alerta += int(reglas.any(axis=1).sum())
            conteos_anio_estado.append(
                elegibles.groupby([elegibles["issue_d"].dt.year, "loan_status"],
                                  dropna=False).size())
        if verbose and (numero + 1) % 10 == 0:
            print(f"  ... {total_filas:,} filas revisadas", flush=True)

    poblacion_elegible = pd.concat(conteos_anio_estado).groupby(level=[0, 1]).sum()
    total_elegibles = int(poblacion_elegible.sum())
    fraccion = min(1.0, tamano_muestra / total_elegibles)


    partes = []
    lector = pd.read_csv(ruta, dtype=tipos, chunksize=tamano_bloque,
                         skipinitialspace=True)
    for numero, bloque in enumerate(lector):
        bloque = _normalizar(bloque)
        elegibles = bloque[bloque["loan_status"].isin(ESTADOS_CONSERVADOS)]
        if len(elegibles):
            partes.append(elegibles.sample(frac=fraccion,
                                           random_state=semilla + numero))
    muestra = pd.concat(partes, ignore_index=True)

    muestra["default"] = muestra["loan_status"].map(MAPA_OBJETIVO)
    muestra["fecha_prestamo"] = muestra["issue_d"]
    muestra["anio"] = muestra["fecha_prestamo"].dt.year
    assert muestra["default"].notna().all(), "Hay un estado sin mapear al objetivo."
    assert muestra["anio"].notna().all(), "Hay un prestamo elegible sin issue_d."

    auditoria = {
        "total_filas": total_filas,
        "total_columnas": len(columnas),
        "columnas": columnas,
        "tipos_lectura": tipos,
        "filas_resumen": filas_resumen,
        "total_elegibles": total_elegibles,
        "fraccion_muestreo": fraccion,
        "estados": estados.astype(int).sort_values(ascending=False),
        "faltantes_globales": faltantes.astype(int),
        "poblacion_anio_estado": poblacion_elegible,
        "alertas": alertas.astype(int).sort_values(ascending=False),
        "filas_con_alerta": filas_con_alerta,
        "valores_policy": sorted(valores_policy),
    }
    return muestra, auditoria


def dividir_por_anio(muestra, corte_validacion, corte_prueba):
    """Particion temporal por anio de otorgamiento. Sin solapamiento."""
    entrenamiento = muestra[muestra["anio"] < corte_validacion].copy()
    validacion = muestra[(muestra["anio"] >= corte_validacion)
                         & (muestra["anio"] < corte_prueba)].copy()
    prueba = muestra[muestra["anio"] >= corte_prueba].copy()
    assert len(entrenamiento) + len(validacion) + len(prueba) == len(muestra)
    return entrenamiento, validacion, prueba
