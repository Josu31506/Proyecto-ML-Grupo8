"""Resolución de rutas desde raíz, notebooks o un subdirectorio del proyecto."""
from pathlib import Path
import os

def raiz_proyecto(inicio=None):
    candidatos = [Path(inicio or Path.cwd()).resolve(), Path(__file__).resolve().parents[1]]
    if os.environ.get("PROYECTO_ML_RAIZ"):
        candidatos.insert(0, Path(os.environ["PROYECTO_ML_RAIZ"]).resolve())
    for base in candidatos:
        for p in [base, *base.parents]:
            if (p / "src/config.py").is_file():
                return p
    raise FileNotFoundError("No se encontró src/config.py; configure PROYECTO_ML_RAIZ.")
