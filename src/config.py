"""Parametros compartidos por todo el proyecto.

Un unico lugar para semillas, rutas y cortes temporales, de modo que el
notebook y los scripts de `src/` produzcan exactamente los mismos numeros.
"""
from pathlib import Path

SEMILLA = 42


TAMANO_BLOQUE = 50_000
TAMANO_MUESTRA = 220_000


ANIO_CORTE_VALIDACION = 2015
ANIO_CORTE_PRUEBA = 2017


UMBRAL_FALTANTES = 95


UMBRAL_FALTANTES_FINAL = 10


UMBRAL_INDICADOR = 0

UMBRAL_REVISION = 12


ESTADOS_CONSERVADOS = ["Fully Paid", "Charged Off"]
MAPA_OBJETIVO = {"Fully Paid": 0, "Charged Off": 1}

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figures"
DIR_SALIDAS = RAIZ / "outputs"


NOMBRES_ARCHIVO = [
    "accepted_2007_to_2018Q4.csv.gz",
    "accepted_2007_to_2018Q4.csv",
]


def localizar_datos(extra=None):
    """Devuelve la ruta al CSV de aceptados buscando en ubicaciones habituales."""
    candidatos = []
    if extra is not None:
        candidatos.append(Path(extra))
    for nombre in NOMBRES_ARCHIVO:
        candidatos += [DIR_DATOS / "raw" / nombre, DIR_DATOS / nombre, RAIZ / nombre,
                       Path.home() / "Downloads" / nombre]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        "No se encontro el dataset. Descarguelo de "
        "https://www.kaggle.com/datasets/wordsforthewise/lending-club y "
        f"coloquelo en {DIR_DATOS / 'raw'}/. Nombres aceptados: {NOMBRES_ARCHIVO}"
    )
