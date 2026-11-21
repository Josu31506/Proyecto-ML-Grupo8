"""Carga común y consulta de la evidencia de los experimentos iniciales."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ANIO_CORTE_PRUEBA, ANIO_CORTE_VALIDACION, SEMILLA,
                     TAMANO_BLOQUE, TAMANO_MUESTRA, localizar_datos)
from .data import dividir_por_anio, escanear_y_muestrear
from .rutas import raiz_proyecto

CONJUNTOS = ("entrenamiento", "validacion", "prueba")
LOGISTICAS = ("logistica_sin_indicadores", "logistica_con_indicadores")


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def cargar_experimento_inicial(ruta_datos=None, tamano_muestra=TAMANO_MUESTRA,
                              modo="consulta", raiz=None, verbose=False):
    """Consulta la auditoría o ejecuta una sola vez el muestreador original.

    Consulta no abre el CSV ni construye particiones: muestra sus tamaños
    históricos. Reproducción devuelve las tablas reales, sin entrenar modelos.
    El redondeo por bloque se conserva; no se recorta la muestra efectiva.
    """
    if modo not in ("consulta", "reproduccion"):
        raise ValueError("Use modo consulta o reproduccion.")
    raiz = raiz_proyecto(raiz)
    if modo == "consulta":
        if tamano_muestra != TAMANO_MUESTRA:
            raise ValueError("La auditoría publicada corresponde al objetivo 220000.")
        auditoria = leer_json(raiz / "outputs/auditoria.json")
        resumen = pd.DataFrame.from_dict(auditoria["particion"], orient="index")
        resumen = resumen.loc[list(CONJUNTOS)].rename(
            columns={"tasa_default": "tasa_incumplimiento"})
        resumen["periodo"] = (resumen.pop("anio_min").astype(str) + "–"
                              + resumen.pop("anio_max").astype(str))
        return {"modo": modo, "raiz": raiz, "ruta": None, "muestra": None,
                "particiones": None, "auditoria": auditoria,
                "columnas_vacias": list(auditoria["columnas_vacias"]),
                "resumen": resumen}

    ruta = localizar_datos(ruta_datos)
    muestra, auditoria = escanear_y_muestrear(
        ruta, tamano_muestra=tamano_muestra, tamano_bloque=TAMANO_BLOQUE,
        semilla=SEMILLA, verbose=verbose)
    partes = dividir_por_anio(muestra, ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA)
    particiones = dict(zip(CONJUNTOS, partes))
    faltantes = auditoria["faltantes_globales"]
    columnas_vacias = faltantes[faltantes == auditoria["total_filas"]].index.tolist()
    resumen = pd.DataFrame([
        {"filas": len(parte), "tasa_incumplimiento": float(parte["default"].mean()),
         "periodo": f"{int(parte['anio'].min())}–{int(parte['anio'].max())}"}
        for parte in partes], index=CONJUNTOS)
    return {"modo": modo, "raiz": raiz, "ruta": ruta, "muestra": muestra,
            "particiones": particiones, "auditoria": auditoria,
            "columnas_vacias": columnas_vacias, "resumen": resumen}


def cargar_referencias(raiz=None):
    """Lee métricas guardadas; nunca ajusta otra regresión logística."""
    raiz = raiz_proyecto(raiz)
    return {"logistica": leer_json(raiz / "outputs/metrics.json"),
            "arbol": leer_json(raiz / "outputs/metrics_arbol.json"),
            "boosting": leer_json(raiz / "outputs/gradient_boosting_220000/metrics_boosting.json")}


def validar_referencias(experimento, referencias):
    """Impide comparar métricas guardadas con otra receta o población."""
    for registro in referencias:
        if (registro["semilla"] != SEMILLA
                or registro["tamano_muestra_objetivo"] != TAMANO_MUESTRA
                or registro["cortes"] != {"validacion": 2015, "prueba": 2017}):
            raise ValueError("La referencia no corresponde al experimento inicial.")
        base = registro["resultados"][LOGISTICAS[0]]
        for nombre, fila in experimento["resumen"].iterrows():
            if (int(fila["filas"]) != base[nombre]["n"]
                    or not np.isclose(fila["tasa_incumplimiento"],
                                      base[nombre]["tasa_default_real"], atol=1e-12, rtol=0)):
                raise ValueError(f"Filas o etiquetas diferentes de la referencia en {nombre}.")


def comprobar_predictoras(datos, raiz=None):
    """Contrasta las 65 características originales, sin ajustar la logística."""
    raiz = raiz_proyecto(raiz)
    decisiones = pd.read_csv(
        raiz / "outputs/decisiones_faltantes_logistica_sin_indicadores.csv", index_col=0)
    columnas = decisiones.index[decisiones["decision"] != "excluir"].tolist()
    if len(columnas) != 65 or set(datos["columnas"]) != set(columnas):
        raise ValueError("Las predictoras no coinciden con las 65 del baseline inicial.")


def rutas_reproduccion(nombre, salida=None, figuras=None):
    """Genera destinos nuevos y rechaza cualquier carpeta ya ocupada."""
    raiz = raiz_proyecto()
    etiqueta = nombre + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    salida = Path(salida) if salida is not None else raiz / "outputs/reproducciones" / etiqueta
    figuras = Path(figuras) if figuras is not None else raiz / "reports/figures/reproducciones" / etiqueta
    destinos = [salida.resolve(), figuras.resolve()]
    protegidos = [raiz / "outputs", raiz / "reports/figures"]
    for destino in destinos:
        if destino in [p.resolve() for p in protegidos]:
            raise ValueError("Use subcarpetas independientes para las reproducciones.")
        if destino.exists() and (not destino.is_dir() or any(destino.iterdir())):
            raise FileExistsError(f"Destino ocupado: {destino}. Use otra carpeta.")
    if (destinos[0] == destinos[1] or destinos[0] in destinos[1].parents
            or destinos[1] in destinos[0].parents):
        raise ValueError("Las carpetas de métricas y figuras deben ser independientes.")
    return salida, figuras
