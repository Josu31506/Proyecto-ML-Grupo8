"""Control y verificación de notebooks; no implementa un pipeline de modelos."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from uuid import uuid4
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

MENSAJE_CONSULTA = (
    "Modo consulta: resultados históricos guardados. "
    "No se ha ejecutado una nueva reproducción."
)
ATOL = 1e-8
RTOL = 1e-6
COLUMNAS = ["Elemento", "Original", "Reproducido", "Diferencia", "Estado"]
CLAVES = (
    "modelo", "experimento", "seleccion", "metodo", "método", "tratamiento",
    "conjunto_variables", "conjunto", "indicadores_faltantes", "configuracion",
    "tipo_configuracion", "anio_validacion", "anio_evaluacion", "anio_final_entrenamiento",
    "anio", "loan_status", "grupo_disponibilidad", "metodo_comparado", "candidatas",
    "referencia", "tipo_referencia", "variables", "tipo", "metrica",
    "variable_modelo", "variable_fuente",
    "umbral_pct", "variable", "variable_26", "variable_59", "corte", "segmento",
    "categoria", "grupo", "max_depth", "min_samples_leaf", "class_weight",
    "learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization",
)


def _operativo(nombre):
    n = str(nombre).lower()
    return any(p in n for p in ("segundo", "tiempo", "duracion", "fecha_ejecucion", "timestamp"))


def _limpiar(valor):
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, Path):
        return str(valor)
    return valor


def _texto(valor):
    texto = json.dumps(valor, ensure_ascii=False, default=_limpiar)
    return texto if len(texto) <= 240 else texto[:237] + "..."


def _fila(elemento, original=None, nuevo=None, diferencia=None, estado="No verificable"):
    return dict(zip(COLUMNAS, [elemento, _texto(original), _texto(nuevo), diferencia, estado]))


def _igual(a, b, atol, rtol):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, Integral) and isinstance(b, Integral):
        return a == b
    if isinstance(a, Real) and isinstance(b, Real):
        return bool(np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    return type(a) is type(b) and a == b


def _aplanar(obj, prefijo=""):
    if isinstance(obj, dict):
        if not obj:
            return {prefijo: {}}
        return {k: v for key, val in obj.items()
                for k, v in _aplanar(val, f"{prefijo}/{key}".strip("/")).items()}
    if isinstance(obj, list):
        if not obj:
            return {prefijo: []}
        return {k: v for i, val in enumerate(obj)
                for k, v in _aplanar(val, f"{prefijo}[{i}]").items()}
    return {prefijo: obj}


def comparar_json(original, reproducido, elemento="JSON", atol=ATOL, rtol=RTOL):
    a, b = _aplanar(original), _aplanar(reproducido)
    filas = []
    for k in sorted(a.keys() | b.keys()):
        if _operativo(k):
            continue
        if k not in a or k not in b:
            filas.append(_fila(f"{elemento}/{k}", a.get(k), b.get(k),
                               "No verificable con la evidencia disponible"))
            continue
        va, vb = a[k], b[k]
        ok = _igual(va, vb, atol, rtol)
        delta = (float(vb) - float(va)
                 if isinstance(va, Real) and isinstance(vb, Real) else None)
        filas.append(_fila(f"{elemento}/{k}", va, vb, delta, "Coincide" if ok else "Difiere"))
    return pd.DataFrame(filas, columns=COLUMNAS)


def comparar_tablas(original, reproducido, elemento="CSV", claves=None, atol=ATOL, rtol=RTOL):
    a, b = original.copy(), reproducido.copy()
    if claves is None:
        claves = [c for c in CLAVES if c in a and c in b]
        claves += [c for c in a if str(c).startswith("Unnamed:") and c in b]
    filas = [_fila(f"{elemento}/filas", len(a), len(b), len(b)-len(a),
                   "Coincide" if len(a) == len(b) else "Difiere")]
    columnas_a = [c for c in a if not _operativo(c)]
    columnas_b = [c for c in b if not _operativo(c)]
    filas.append(_fila(f"{elemento}/columnas", columnas_a, columnas_b, None,
                       "Coincide" if set(columnas_a) == set(columnas_b) else "Difiere"))
    if claves:
        if any(c not in a or c not in b for c in claves) or a.duplicated(claves).any() or b.duplicated(claves).any():
            filas.append(_fila(f"{elemento}/alineación", diferencia="Claves ausentes o duplicadas: no se presupone correspondencia."))
            return pd.DataFrame(filas, columns=COLUMNAS)
        def indexar(df):
            copia = df.copy()
            copia.index = copia[claves].apply(lambda row: json.dumps(
                [None if pd.isna(v) else _limpiar(v) for v in row], ensure_ascii=False), axis=1)
            return copia.sort_index()
        a, b = indexar(a), indexar(b)
        if set(a.index) != set(b.index):
            filas.append(_fila(f"{elemento}/claves", a.index.tolist(), b.index.tolist(),
                               "Diferencias en observaciones de la tabla", "Difiere"))
            return pd.DataFrame(filas, columns=COLUMNAS)
    elif len(a) != len(b):
        return pd.DataFrame(filas, columns=COLUMNAS)
    else:
        a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    for col in sorted(set(columnas_a) & set(columnas_b)):
        av, bv = a[col].tolist(), b[col].tolist()
        ok, delta = [], []
        for x, y in zip(av, bv):
            x = None if pd.isna(x) else _limpiar(x)
            y = None if pd.isna(y) else _limpiar(y)
            ok.append(_igual(x, y, atol, rtol))
            if isinstance(x, Real) and isinstance(y, Real):
                delta.append(abs(float(y)-float(x)))
        filas.append(_fila(f"{elemento}/{col}", av, bv,
                           max(delta, default=0.0) if all(ok) else f"{len(ok)-sum(ok)} diferencias; máx. absoluta={max(delta, default=0.0):.12g}",
                           "Coincide" if all(ok) else "Difiere"))
    return pd.DataFrame(filas, columns=COLUMNAS)


def comparar_archivos(original, reproducido, claves=None, atol=ATOL, rtol=RTOL):
    a, b = Path(original), Path(reproducido)
    if not a.is_file() or not b.is_file():
        return pd.DataFrame([_fila(a.name, str(a), str(b),
            "No verificable con la evidencia disponible: falta " + str(a if not a.is_file() else b))])
    if a.suffix == ".csv":
        return comparar_tablas(pd.read_csv(a), pd.read_csv(b), a.name, claves, atol, rtol)
    if a.suffix == ".json":
        return comparar_json(json.loads(a.read_text(encoding="utf-8-sig")),
                             json.loads(b.read_text(encoding="utf-8-sig")), a.name, atol, rtol)
    return pd.DataFrame([_fila(a.name, diferencia="No verificable: se comparan CSV y JSON, no identidad visual ni objetos entrenados.")])


class SesionNotebook:
    def __init__(self, raiz, nombre, reproducir=False, ruta_datos=None, confirmar_test=False):
        self.raiz = Path(raiz).resolve()
        self.nombre = nombre
        self.requiere_test = nombre.endswith(("_evaluacion", "_evaluacion_final"))
        self.reproducir = reproducir
        self.ruta_datos = ruta_datos
        self.confirmar_test = confirmar_test
        self.base = (self.raiz / "outputs/reproducciones").resolve()
        self.salida = None
        self._figuras = {}
        self._tablas = 0
        self.manifiesto = {
            "notebook": nombre, "estado": "No verificado", "reproduccion_ejecutada": False,
            "tolerancias": {"absoluta": ATOL, "relativa": RTOL, "enteros_y_categorias": "exactos"},
            "limitaciones": ["Coincidir en métricas no demuestra identidad de muestras.",
                "Sin huellas históricas de filas/dataset, la identidad no es verificable.",
                "Los tiempos de cómputo se registran pero no se exige su igualdad."],
            "entradas_historicas": {}, "datos": None, "configuracion": {},
        }
        if reproducir:
            if self.requiere_test and not confirmar_test:
                raise PermissionError("Para reproducir el test se requiere CONFIRMAR_TEST = True.")
            self.salida = self.base / (nombre + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8])
            self.salida.mkdir(parents=True, exist_ok=False)
            self.manifiesto["inicio_utc"] = datetime.now(timezone.utc).isoformat()
            self.manifiesto["versiones"] = {p: importlib.metadata.version(p)
                                          for p in ("numpy", "pandas", "scikit-learn")}
            self.manifiesto["implementacion_sha256"] = {
                str(p.relative_to(self.raiz)): hashlib.sha256(p.read_bytes()).hexdigest()
                for carpeta in ("src", "notebooks") for p in (self.raiz / carpeta).glob("*")
                if p.suffix in (".py", ".ipynb")
            }
            if self.requiere_test:
                self.manifiesto["alcance_test"] = "Reproducción retrospectiva del test ya utilizado; no es una nueva evaluación independiente."
            self._guardar_manifiesto()
            print("Modo reproducción: se ejecutará el procedimiento original al correr las celdas.")
            print("Salidas independientes:", self.salida)
        else:
            print(MENSAJE_CONSULTA)

    def exigir(self):
        if not self.reproducir or self.salida is None:
            raise RuntimeError("Esta operación requiere REPRODUCIR = True desde el inicio.")
        if self.requiere_test and not self.confirmar_test:
            raise PermissionError("Falta CONFIRMAR_TEST = True.")
        if not self.salida.resolve().is_relative_to(self.base):
            raise RuntimeError("La salida debe pertenecer a outputs/reproducciones/.")

    def requerir(self, rutas):
        faltan = [str(Path(p)) for p in rutas if not Path(p).is_file()]
        if faltan:
            raise FileNotFoundError("Faltan artefactos históricos; no se inicia entrenamiento:\n" + "\n".join(faltan))
        for p in rutas:
            p = Path(p)
            self.manifiesto["entradas_historicas"][str(p.relative_to(self.raiz))] = hashlib.sha256(p.read_bytes()).hexdigest()

    def registrar(self, configuracion=None, datos=None):
        self.exigir()
        if configuracion:
            self.manifiesto["configuracion"].update(configuracion)
        if datos is not None:
            p = Path(datos)
            h = hashlib.sha256()
            with p.open("rb") as f:
                for bloque in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    h.update(bloque)
            self.manifiesto["datos"] = {"archivo": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
        self._guardar_manifiesto()

    def _guardar_manifiesto(self):
        self.exigir()
        (self.salida / "manifiesto.json").write_text(json.dumps(self.manifiesto, ensure_ascii=False,
            indent=2, default=_limpiar), encoding="utf-8")

    def guardar_figura(self, figura, etiqueta):
        self.exigir()
        self._figuras[etiqueta] = self._figuras.get(etiqueta, 0) + 1
        nombre = f"{etiqueta}_{self._figuras[etiqueta]:02d}.png"
        figura.savefig(self.salida / nombre, dpi=150, bbox_inches="tight")

    def mostrar(self, *objetos, **kwargs):
        from IPython.display import display
        for obj in objetos:
            if self.reproducir and isinstance(obj, (pd.DataFrame, pd.Series)):
                self.exigir()
                if len(obj) <= 2000:
                    self._tablas += 1
                    obj.to_csv(self.salida / f"tabla_descriptiva_{self._tablas:03d}.csv")
        display(*objetos, **kwargs)

    def finalizar(self, pares, configuracion=None, datos=None, completa=True, comparador=None):
        if not self.reproducir:
            return pd.DataFrame([_fila("Reproducción", diferencia="Solo se consultaron resultados históricos.", estado="No verificado")])
        self.registrar(configuracion, datos)
        funcion_comparar = comparador or comparar_archivos
        tablas = [funcion_comparar(a, b) for a, b in pares]
        for relativo, esperado in self.manifiesto["entradas_historicas"].items():
            p = self.raiz / relativo
            actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            tablas.append(pd.DataFrame([_fila("Integridad histórica/" + relativo,
                esperado, actual, None, "Coincide" if esperado == actual else "Difiere")]))
        tablas.append(pd.DataFrame([_fila("Identidad del dataset y filas históricas", diferencia=
            "No verificable con la evidencia disponible: los archivos históricos no siempre guardan las huellas de filas/dataset.")]))
        if not completa:
            tablas.append(pd.DataFrame([_fila("Alcance", diferencia="Modo de prueba: no reproduce la búsqueda completa.")]))
        tabla = pd.concat(tablas, ignore_index=True)
        self.manifiesto.update(reproduccion_ejecutada=True, completa=completa,
            estado="Difiere" if (tabla.Estado == "Difiere").any() else
                   ("No verificable" if (tabla.Estado == "No verificable").any() else "Coincide"),
            resumen_verificacion=tabla.Estado.value_counts().to_dict(),
            fin_utc=datetime.now(timezone.utc).isoformat())
        tabla.to_csv(self.salida / "verificacion.csv", index=False)
        self._guardar_manifiesto()
        return tabla


def mostrar_resultados_guardados(raiz, indice, celda):
    """Lee el índice de outputs y presenta los archivos canónicos referenciados."""
    from IPython.display import HTML, Image, display
    raiz = Path(raiz).resolve()
    p = Path(indice)
    if not p.is_file():
        print("No verificable con la evidencia disponible: falta", p)
        return
    referencias = json.loads(p.read_text(encoding="utf-8"))["celdas"].get(str(celda), [])
    if not referencias:
        print(f"No verificable con la evidencia disponible: no hay salida histórica guardada para la celda {celda}.")
    for referencia in referencias:
        if referencia.get("descripcion"):
            print(referencia["descripcion"])
        if referencia["tipo"] == "nota":
            continue
        rutas = [raiz / nombre for nombre in referencia.get("archivos", [referencia.get("archivo")])]
        faltantes = [str(r) for r in rutas if not r.is_file()]
        if faltantes:
            print("No verificable con la evidencia disponible: falta " + ", ".join(faltantes))
            continue
        archivo = rutas[0]
        tipo = referencia["tipo"]
        if tipo == "imagen":
            display(Image(filename=str(archivo)))
        elif tipo == "html":
            display(HTML(archivo.read_text(encoding="utf-8")))
        elif tipo == "csv":
            display(pd.read_csv(archivo))
        elif tipo == "json":
            datos = json.loads(archivo.read_text(encoding="utf-8"))
            vista = referencia.get("vista")
            if vista == "estados":
                tabla = pd.Series(datos["estados"], name="cantidad").to_frame()
                tabla["porcentaje"] = tabla["cantidad"] / datos["total_filas"] * 100
                display(tabla.sort_values("cantidad", ascending=False))
            elif vista == "objetivo":
                tabla = pd.Series({k: datos["estados"][k] for k in ["Fully Paid", "Charged Off"]}, name="cantidad").to_frame()
                tabla["porcentaje"] = tabla["cantidad"] / tabla["cantidad"].sum() * 100
                display(tabla)
            elif vista == "particiones":
                display(pd.DataFrame.from_dict(datos["particion"], orient="index"))
            elif vista == "umbrales":
                display(pd.DataFrame([{"modelo": m, "umbral_validacion": d["validacion"]["umbral"]}
                                      for m, d in datos["resultados"].items()]))
            else:
                display({k: datos[k] for k in referencia["campos"]}
                        if "campos" in referencia else datos)
        elif tipo == "benchmark_fases":
            fases = []
            for ruta in rutas:
                dato = json.loads(ruta.read_text(encoding="utf-8"))
                for fase in dato.get("fases", []):
                    fases.append({"fase": fase["fase"], "metodo": dato["metodo"],
                                  "segundos": fase["segundos"], "pico_GiB": fase["pico_rss_bytes"] / 1024**3})
            tabla = pd.DataFrame(fases).groupby(["fase", "metodo"]).agg(
                segundos_promedio=("segundos", "mean"), pico_GiB_promedio=("pico_GiB", "mean"))
            display(tabla.round(2).reset_index())
        else:
            raise ValueError(f"Tipo de resultado guardado desconocido: {tipo}")
