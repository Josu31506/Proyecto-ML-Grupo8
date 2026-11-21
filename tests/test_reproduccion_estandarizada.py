"""Controles de los once notebooks y comparadores con evidencia sintética."""
from contextlib import ExitStack, redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import IPython.display
import matplotlib.pyplot as plt
import nbformat
import pandas as pd

from src import etapas_finales, modelado
from src.reproduccion_notebooks import (
    ATOL, MENSAJE_CONSULTA, SesionNotebook, comparar_archivos, comparar_json, comparar_tablas,
    mostrar_resultados_guardados,
)


class ReproduccionEstandarizadaTests(unittest.TestCase):
    RAIZ = Path(__file__).resolve().parents[1]

    def ejecutar_notebook(self, archivo, directorio=None, reproducir=False):
        env = {"__name__": "__consulta__"}
        salida = io.StringIO()
        with patch("pathlib.Path.cwd", return_value=directorio or self.RAIZ), redirect_stdout(salida):
            for celda in json.loads(archivo.read_text(encoding="utf-8"))["cells"]:
                if celda["cell_type"] != "code":
                    continue
                codigo = "".join(celda["source"])
                if reproducir:
                    codigo = codigo.replace("REPRODUCIR = False", "REPRODUCIR = True")
                exec(compile(codigo, archivo.name, "exec"), env)
        return env, salida.getvalue()

    def bloquear(self, stack):
        for target in [
            "src.config.localizar_datos", "src.experimento_inicial.localizar_datos",
            "src.experimento_inicial.escanear_y_muestrear", "src.data.escanear_y_muestrear",
            "src.modelado.cargar_datos_modelado", "src.etapas_finales.cargar_desarrollo",
            "src.etapas_finales.auditar_y_cargar_test", "src.etapas_finales.correr_corte",
            "src.etapas_finales.entrenar_final_y_evaluar", "src.analisis_errores.ejecutar_analisis",
            "src.auditoria_memoria.ejecutar", "src.entrenamiento_inicial.ejecutar",
            "src.comparacion_variables.ejecutar", "src.comparacion_imputacion.ejecutar",
            "src.comparacion_lineales.ejecutar", "src.comparacion_umbrales.ejecutar",
            "src.sensibilidad_faltantes.ejecutar", "sklearn.pipeline.Pipeline.fit",
            "sklearn.pipeline.Pipeline.fit_transform", "sklearn.compose.ColumnTransformer.fit_transform",
            "sklearn.model_selection.RandomizedSearchCV.fit", "sklearn.linear_model.LogisticRegression.fit",
            "sklearn.svm.LinearSVC.fit", "sklearn.tree.DecisionTreeClassifier.fit",
            "sklearn.ensemble.HistGradientBoostingClassifier.fit", "sklearn.calibration.calibration_curve",
            "pandas.DataFrame.to_csv", "pathlib.Path.mkdir", "pathlib.Path.write_text",
            "matplotlib.figure.Figure.savefig",
        ]:
            stack.enter_context(patch(target, side_effect=AssertionError(target)))
        stack.enter_context(patch("IPython.display.display"))
        stack.enter_context(patch("IPython.display.Image"))
        stack.enter_context(patch("matplotlib.pyplot.show"))
        leer = pd.read_csv

        def leer_agregado(path, *args, **kwargs):
            nombre = Path(path).name
            if nombre.startswith(("accepted_", "rejected_", "predicciones")):
                raise AssertionError("Consulta abrió datos por préstamo.")
            return leer(path, *args, **kwargs)

        stack.enter_context(patch("pandas.read_csv", side_effect=leer_agregado))

    def test_once_formatos_y_controles_visibles(self):
        archivos = sorted((self.RAIZ / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(archivos), 11)
        for archivo in archivos:
            with self.subTest(archivo=archivo.name):
                nb = nbformat.read(archivo, as_version=4)
                nbformat.validate(nb)
                celdas = [c.source for c in nb.cells if c.cell_type == "code"]
                self.assertEqual(sum("REPRODUCIR = False" in c for c in celdas), 1)
                self.assertLess(next(i for i, c in enumerate(nb.cells)
                                     if c.cell_type == "code" and "REPRODUCIR = False" in c.source), 7)
                self.assertFalse(any("REGENERAR_" in c for c in celdas))
                texto = "\n".join(c.source for c in nb.cells)
                for requerido in ("Control de ejecución: consulta o reproducción", "Verificación de reproducibilidad"):
                    self.assertIn(requerido, texto)
                self.assertIn("REPRODUCIR = True", texto)
                self.assertIn("RUTA_DATOS", texto)
                for codigo in celdas:
                    compile(codigo, archivo.name, "exec")

    def test_consulta_once_notebooks_sin_dataset_sin_ajustes_sin_escrituras(self):
        for archivo in sorted((self.RAIZ / "notebooks").glob("*.ipynb")):
            for directorio in (self.RAIZ, self.RAIZ / "notebooks"):
                with self.subTest(archivo=archivo.name, cwd=directorio), ExitStack() as stack:
                    self.bloquear(stack)
                    env, stdout = self.ejecutar_notebook(archivo, directorio)
                    self.assertIn(MENSAJE_CONSULTA, stdout)
                    self.assertFalse(env["REPRODUCIR"])
                    self.assertIsNone(env["sesion"].salida)
                    esperados = {"No verificado"}
                    if archivo.name.startswith("05") and not (self.RAIZ / "outputs/svm_muestra/metricas_svm_muestra.csv").is_file():
                        esperados.add("No verificable")
                    self.assertEqual(set(env["verificacion"].Estado), esperados)
                    plt.close("all")

    def test_reproduccion_de_evaluacion_no_requiere_confirmacion_adicional(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            raiz = Path(temporal)
            sesion = SesionNotebook(raiz, "11_evaluacion", True)
            self.assertTrue(sesion.requiere_test)
            self.assertTrue(sesion.salida.is_relative_to(raiz / "outputs/reproducciones"))
            self.assertIn("alcance_test", sesion.manifiesto)
            sesion.exigir()

    def test_consulta_usa_indices_en_outputs_y_reutiliza_resultados_originales(self):
        self.assertFalse((self.RAIZ / "notebooks/evidencia_historica").exists())
        for carpeta in ("exploracion_inicial", "exploracion_completa"):
            with self.subTest(carpeta=carpeta):
                indice = self.RAIZ / "outputs" / carpeta / "consulta.json"
                datos = json.loads(indice.read_text(encoding="utf-8"))
                referencias = [r for celda in datos["celdas"].values() for r in celda if r["tipo"] != "nota"]
                self.assertTrue(referencias)
                for ref in referencias:
                    self.assertNotIn("data", ref)
                    for relativo in ref.get("archivos", [ref.get("archivo")]):
                        p = self.RAIZ / relativo
                        self.assertTrue(p.is_file(), relativo)
                        self.assertTrue(p.is_relative_to(self.RAIZ / "outputs") or
                                        p.is_relative_to(self.RAIZ / "reports/figures"))
                for relativo, fuente in datos["extraidos"].items():
                    actual = hashlib.sha256((self.RAIZ / relativo).read_bytes()).hexdigest()
                    self.assertEqual(actual, fuente["sha256"])
                self.assertTrue(any(ref.get("archivo") == "outputs/auditoria.json" for ref in referencias))
                if carpeta == "exploracion_inicial":
                    for existente in ("outputs/metrics.json", "outputs/coeficientes_baseline.csv",
                                      "reports/figures/exploracion/correlacion_antes_A.png"):
                        self.assertTrue(any(ref.get("archivo") == existente for ref in referencias), existente)
                    self.assertFalse((indice.parent / "metrics.json").exists())
                    self.assertFalse((indice.parent / "coeficientes_baseline.csv").exists())

    def test_lector_consulta_lee_la_fuente_actual_y_avisa_si_falta_sin_recalcular(self):
        with tempfile.TemporaryDirectory() as temporal:
            r = Path(temporal)
            (r / "outputs").mkdir()
            datos = r / "outputs/metricas.json"
            indice = r / "outputs/consulta.json"
            indice.write_text(json.dumps({"celdas": {"1": [
                {"tipo": "json", "archivo": "outputs/metricas.json"},
                {"tipo": "csv", "archivo": "outputs/ausente.csv"}]}}), encoding="utf-8")
            for valor in (.7123456789, .7654321098):
                datos.write_text(json.dumps({"roc_auc": valor}), encoding="utf-8")
                stdout = io.StringIO()
                with ExitStack() as stack, redirect_stdout(stdout):
                    self.bloquear(stack)
                    mostrar = stack.enter_context(patch("IPython.display.display"))
                    mostrar_resultados_guardados(r, indice, 1)
                    mostrar.assert_called_once_with({"roc_auc": valor})
                self.assertIn("ausente.csv", stdout.getvalue())
                self.assertIn("No verificable", stdout.getvalue())

    def test_comparacion_csv_alinea_anios_sin_redondear(self):
        original = pd.DataFrame({"modelo": ["m", "m"], "anio_validacion": [2013, 2014], "ROC_AUC": [.700000001, .71], "n": [40, 50]})
        copia = original.iloc[::-1].copy()
        copia.loc[0, "ROC_AUC"] += ATOL / 2
        self.assertEqual(set(comparar_tablas(original, copia).Estado), {"Coincide"})
        copia.loc[0, "ROC_AUC"] += .00002
        self.assertIn("Difiere", set(comparar_tablas(original, copia).Estado))
        copia = original.copy()
        copia.loc[0, "n"] += 1
        self.assertIn("Difiere", set(comparar_tablas(original, copia).Estado))
        estados = pd.DataFrame({"anio": [2017, 2017], "loan_status": ["Fully Paid", "Charged Off"], "cantidad": [80, 20]})
        self.assertEqual(set(comparar_tablas(estados, estados.iloc[::-1]).Estado), {"Coincide"})

    def test_claves_duplicadas_y_faltantes_no_producen_exito(self):
        a = pd.DataFrame({"modelo": ["a", "a"], "roc_auc": [.7, .71]})
        self.assertIn("No verificable", set(comparar_tablas(a, a).Estado))
        b = a.copy()
        b.loc[1, "modelo"] = "b"
        self.assertIn("Difiere", set(comparar_tablas(b.iloc[:1], b).Estado))
        tabla = comparar_archivos("ausente.csv", "tambien_ausente.csv")
        self.assertEqual(set(tabla.Estado), {"No verificable"})

    def test_json_configuracion_semillas_y_sin_tolerancia_para_enteros(self):
        base = {"semilla": 42, "particiones": [2013, 2014], "metodo": "Sigmoide", "ROC_AUC": .7123}
        self.assertEqual(set(comparar_json(base, dict(base)).Estado), {"Coincide"})
        self.assertIn("Difiere", set(comparar_json(base, dict(base, semilla=43)).Estado))
        self.assertIn("Difiere", set(comparar_json({"n": 1000000000}, {"n": 1000000001}).Estado))
        self.assertIn("No verificable", set(comparar_json(base, {}).Estado))

    def test_sesion_consulta_no_crea_carpeta_y_faltante_es_explicito(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            r = Path(temporal)
            sesion = SesionNotebook(r, "01_baseline")
            self.assertIsNone(sesion.salida)
            self.assertFalse((r / "outputs").exists())
            with self.assertRaisesRegex(FileNotFoundError, "falta.csv"):
                sesion.requerir([r / "falta.csv"])
            self.assertEqual(set(sesion.finalizar([]).Estado), {"No verificado"})

    def test_sesion_reproduce_con_manifiesto_y_comparacion_en_carpeta_nueva(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            r = Path(temporal)
            original = r / "historico.csv"
            a = pd.DataFrame({"modelo": ["m"], "ROC_AUC": [.7]})
            a.to_csv(original, index=False)
            contenido = original.read_bytes()
            primera = SesionNotebook(r, "01_baseline", True)
            segunda = SesionNotebook(r, "01_baseline", True)
            self.assertNotEqual(primera.salida, segunda.salida)
            self.assertTrue(primera.salida.is_relative_to(r / "outputs/reproducciones"))
            primera.requerir([original])
            a.to_csv(primera.salida / "nuevo.csv", index=False)
            tabla = primera.finalizar([(original, primera.salida / "nuevo.csv")], {"semilla": 42})
            self.assertIn("Coincide", set(tabla.Estado))
            self.assertIn("No verificable", set(tabla.Estado))
            manifest = json.loads((primera.salida / "manifiesto.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["reproduccion_ejecutada"])
            self.assertEqual(manifest["configuracion"]["semilla"], 42)
            self.assertEqual(original.read_bytes(), contenido)
            self.assertTrue((primera.salida / "verificacion.csv").is_file())

    def test_modo_prueba_no_equivale_a_reproduccion_completa(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            sesion = SesionNotebook(temporal, "03_boosting", True)
            tabla = sesion.finalizar([], completa=False)
            self.assertIn("Modo de prueba", tabla.Diferencia.str.cat(sep=" "))
            manifest = json.loads((sesion.salida / "manifiesto.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["completa"])
            self.assertEqual(manifest["estado"], "No verificable")

    def test_sensibilidad_y_lineales_reciben_resultados_de_la_misma_sesion(self):
        for nombre, esperado in [("08_umbrales_sensibilidad.ipynb", 'referencias=FUENTE_UMBRALES'),
                                 ("07_comparacion_variables_imputacion.ipynb", 'referencias=SALIDA / "comparacion_variables_temporal"')]:
            nb = nbformat.read(self.RAIZ / "notebooks" / nombre, as_version=4)
            codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
            self.assertIn(esperado, codigo)
            if nombre.startswith("08"):
                self.assertIn('FUENTE_UMBRALES = SALIDA / "comparacion_umbrales_faltantes"', codigo)


if __name__ == "__main__":
    unittest.main()
