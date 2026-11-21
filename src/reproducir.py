"""Ejecuta experimentos por separado sin sobrescribir evidencia histórica."""
from pathlib import Path
from datetime import datetime
import importlib
import json
from .rutas import raiz_proyecto

EXPERIMENTOS = {'auditoria': 'auditoria_memoria', 'entrenamiento_inicial': 'entrenamiento_inicial', 'variables': 'comparacion_variables', 'imputacion': 'comparacion_imputacion', 'lineales': 'comparacion_lineales', 'umbrales': 'comparacion_umbrales', 'sensibilidad': 'sensibilidad_faltantes'}

def ejecutar_experimento(nombre, salida=None, referencias=None, ruta_datos=None):
    if nombre not in EXPERIMENTOS:
        raise ValueError(f"Experimento desconocido: {nombre}")
    raiz = raiz_proyecto()
    carpeta = Path(salida) if salida else raiz / "outputs/reproducciones" / (nombre + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    carpeta.mkdir(parents=True, exist_ok=False)
    (carpeta / "procedencia.json").write_text(json.dumps({"experimento": nombre, "modulo": "src." + EXPERIMENTOS[nombre]}, ensure_ascii=False, indent=2), encoding="utf-8")
    modulo = importlib.import_module("src." + EXPERIMENTOS[nombre])
    kwargs = {"ruta_datos": ruta_datos}
    if nombre in ("sensibilidad", "lineales"):
        kwargs["referencias"] = referencias
    resultado = modulo.ejecutar(raiz, carpeta, **kwargs)
    return carpeta, resultado
