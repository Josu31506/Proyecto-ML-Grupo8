"""Compara únicamente el tiempo de carga directa y por bloques."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd


def huella_tabla(tabla):
    """Crea una huella para comprobar que los métodos cargaron los mismos datos."""
    huella = hashlib.sha256()
    for inicio in range(0, len(tabla), 50_000):
        bloque_hash = pd.util.hash_pandas_object(
            tabla.iloc[inicio:inicio + 50_000], index=False
        )
        huella.update(bloque_hash.to_numpy().tobytes())
    return huella.hexdigest()


def ejecutar(config, metodo, carpeta):
    """Lee el archivo usando un método y registra su tiempo de carga."""
    tipos = {}
    for columna, dato in config["esquema"].items():
        if dato["tipo"] == "category":
            tipos[columna] = pd.CategoricalDtype(dato["categorias"])
        else:
            tipos[columna] = dato["tipo"]

    opciones = {
        "usecols": list(tipos),
        "dtype": tipos,
        "skipinitialspace": True,
    }

    inicio = time.perf_counter()
    if metodo == "directa":
        fuente = pd.read_csv(config["archivo"], **opciones)
        filas_fuente = len(fuente)
        datos = fuente.loc[
            fuente["loan_status"].isin(["Fully Paid", "Charged Off"])
        ].copy()
        del fuente
    else:
        partes = []
        filas_fuente = 0
        with pd.read_csv(
            config["archivo"],
            chunksize=config["bloque"],
            **opciones,
        ) as lector:
            for bloque in lector:
                filas_fuente += len(bloque)
                partes.append(
                    bloque.loc[
                        bloque["loan_status"].isin(["Fully Paid", "Charged Off"])
                    ].copy()
                )
        datos = pd.concat(partes, ignore_index=True)
        del partes, bloque
    segundos = time.perf_counter() - inicio

    # Estas comprobaciones ocurren después del cronómetro: no forman parte
    # del tiempo de carga que estamos comparando.
    assert filas_fuente == config["filas_fuente_esperadas"]
    resultado = {
        "metodo": metodo,
        "segundos_carga": segundos,
        "filas_fuente": filas_fuente,
        "filas_elegibles": len(datos),
        "huella_datos": huella_tabla(datos),
    }
    (carpeta / "resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{metodo}: carga y filtro en {segundos:.2f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compara el tiempo de carga directa y por bloques."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("metodo", choices=["directa", "bloques"])
    parser.add_argument("carpeta", type=Path)
    argumentos = parser.parse_args()
    config = json.loads(argumentos.config.read_text(encoding="utf-8"))
    ejecutar(config, argumentos.metodo, argumentos.carpeta)


if __name__ == "__main__":
    main()
