"""Verificación de observaciones del experimento inicial sin entrenar modelos."""
import hashlib
import json
import numpy as np
import pandas as pd
from .config import ESTADOS_CONSERVADOS, SEMILLA, TAMANO_BLOQUE
from .data import dividir_por_anio


def verificar_muestra(ruta, muestra, auditoria, objetivo, referencias):
    if objetivo != 220000:
        raise ValueError("La comparación oficial requiere objetivo 220000.")
    claves = ["id", "loan_status", "issue_d"]
    fraccion = min(1.0, objetivo / auditoria["total_elegibles"])
    seleccionadas = []
    huella_fuente = hashlib.sha256()
    total = 0
    with pd.read_csv(ruta, usecols=claves, dtype="string", chunksize=TAMANO_BLOQUE,
                     skipinitialspace=True) as lector:
        for numero, bloque in enumerate(lector):
            total += len(bloque)
            valores = pd.util.hash_pandas_object(bloque, index=False).to_numpy()
            huella_fuente.update(valores.tobytes())
            elegibles = bloque.loc[bloque.loan_status.isin(ESTADOS_CONSERVADOS)]
            if len(elegibles):
                seleccionadas.append(elegibles.sample(frac=fraccion, random_state=SEMILLA + numero))
    referencia = pd.concat(seleccionadas, ignore_index=True)
    actual = muestra[claves].reset_index(drop=True).copy()
    referencia["issue_d"] = pd.to_datetime(referencia["issue_d"], format="%b-%Y", errors="raise")
    for c in ["id", "loan_status"]:
        if not referencia[c].astype("string").equals(actual[c].astype("string")):
            raise ValueError(f"Observaciones u orden distintos en {c}.")
    if not np.array_equal(referencia.issue_d.to_numpy(), actual.issue_d.to_numpy()):
        raise ValueError("Fechas distintas en la selección reconstruida.")
    if len(muestra) != 220003 or total != auditoria["total_filas"]:
        raise ValueError("El tamaño efectivo o el recorrido no coincide con el experimento inicial.")
    partes = dict(zip(["entrenamiento", "validacion", "prueba"], dividir_por_anio(muestra, 2015, 2017)))
    for registro in referencias:
        if registro["semilla"] != SEMILLA or registro["tamano_muestra_objetivo"] != objetivo:
            raise ValueError("La referencia usa otra semilla o tamaño objetivo.")
        if registro["cortes"] != {"validacion": 2015, "prueba": 2017}:
            raise ValueError("La referencia usa otros periodos.")
        base = registro["resultados"]["logistica_sin_indicadores"]
        for nombre, parte in partes.items():
            if len(parte) != base[nombre]["n"] or not np.isclose(parte.default.mean(), base[nombre]["tasa_default_real"], atol=1e-12, rtol=0):
                raise ValueError(f"Filas o etiquetas diferentes de la referencia en {nombre}.")
    huellas = {}
    for nombre, parte in partes.items():
        ids = parte.id.astype("string").to_json(orient="values", force_ascii=False)
        huellas[nombre] = hashlib.sha256(ids.encode("utf-8")).hexdigest()
    return {"semilla": SEMILLA, "bloque": TAMANO_BLOQUE, "filas_muestra": len(muestra),
            "filas_fuente": total, "fuente": str(ruta.name),
            "huella_identidades_fuente": huella_fuente.hexdigest(), "huellas_ids_por_particion": huellas,
            "particiones": {n: len(p) for n, p in partes.items()},
            "verificacion": "reconstrucción independiente por IDs, estados y fechas; filas y prevalencias de referencias",
            "limite": "Las métricas históricas no guardan IDs ni hash de la fuente. La coincidencia histórica se sustenta en la receta y sus comprobaciones; la nueva comparación recalcula modelos sobre los mismos objetos."}
