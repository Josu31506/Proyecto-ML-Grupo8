"""Compara el tiempo de creación de un DataFrame con dos esquemas de tipos."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def ejecutar(config, esquema_nombre, carpeta):
    esquema = config["esquemas"][esquema_nombre]
    tipos = {}
    for columna, dato in esquema.items():
        if dato["tipo"] == "category":
            tipos[columna] = pd.CategoricalDtype(dato["categorias"])
        else:
            tipos[columna] = dato["tipo"]

    inicio = time.perf_counter()
    tabla = pd.read_csv(
        config["archivo"],
        usecols=config["columnas"],
        dtype=tipos,
        skipinitialspace=True,
    )
    segundos = time.perf_counter() - inicio

    # Se registran estas propiedades después del cronómetro: el tiempo reportado
    # corresponde solo a la construcción del DataFrame mediante read_csv.
    resultado = {
        "esquema": esquema_nombre,
        "segundos_read_csv": segundos,
        "filas": int(len(tabla)),
        "columnas": int(tabla.shape[1]),
        "memoria_dataframe_bytes": int(
            tabla.memory_usage(index=True, deep=True).sum()
        ),
    }
    (carpeta / "resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{esquema_nombre}: {len(tabla):,} filas, "
        f"{tabla.shape[1]} columnas en {segundos:.2f} s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compara tiempos de read_csv con tipos originales y reducidos."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("esquema", choices=["originales", "reducidos"])
    parser.add_argument("carpeta", type=Path)
    argumentos = parser.parse_args()
    config = json.loads(argumentos.config.read_text(encoding="utf-8"))
    ejecutar(config, argumentos.esquema, argumentos.carpeta)


if __name__ == "__main__":
    main()
