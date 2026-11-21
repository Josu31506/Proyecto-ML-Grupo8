"""Preserva el experimento de muestra con datos sintéticos pequeños."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import LinearSVC

from src import svm_muestra as svm
from src.experimentos import preparar_experimentos
from src.experimento_inicial import cargar_experimento_inicial
from src.reproduccion_notebooks import SesionNotebook, comparar_archivos


class SvmMuestraTests(unittest.TestCase):
    RAIZ = Path(__file__).resolve().parents[1]

    def experimento_sintetico(self):
        n = 36
        fechas = pd.to_datetime(["2014-01-01"] * 20 + ["2015-01-01"] * 8 + ["2017-01-01"] * 8)
        muestra = pd.DataFrame({
            "id": [str(i) for i in range(n)], "issue_d": fechas, "fecha_prestamo": fechas,
            "anio": fechas.year, "default": [i % 2 for i in range(n)],
            "loan_status": ["Fully Paid" if i % 2 == 0 else "Charged Off" for i in range(n)],
            "loan_amnt": [1000. + 500 * (i % 2) + i for i in range(n)],
            "annual_inc": [np.nan if i % 7 == 0 else 50000. + 100 * i for i in range(n)],
            "home_ownership": ["OWN" if i % 3 else "RENT" for i in range(n)],
            "earliest_cr_line": pd.to_datetime(["2000-01-01"] * n), "term": [36.] * n,
        })
        partes = {"entrenamiento": muestra.iloc[:20].copy(), "validacion": muestra.iloc[20:28].copy(),
                  "prueba": muestra.iloc[28:].copy()}
        return {"muestra": muestra, "particiones": partes, "columnas_vacias": [],
                "auditoria": {"total_filas": n, "total_elegibles": n, "fraccion_muestreo": 1.0}}

    def ejecutar_notebook(self, raiz, reproducir=False):
        nb = json.loads((self.RAIZ / "notebooks/05_svm_muestra.ipynb").read_text(encoding="utf-8"))
        env = {"__name__": "__notebook__"}
        with patch("pathlib.Path.cwd", return_value=raiz), redirect_stdout(io.StringIO()):
            for celda in nb["cells"]:
                if celda["cell_type"] == "code":
                    codigo = "".join(celda["source"])
                    if reproducir:
                        codigo = codigo.replace("REPRODUCIR = False", "REPRODUCIR = True")
                    exec(compile(codigo, "05_svm_muestra.ipynb", "exec"), env)
        return env

    def test_modelos_conservan_parametros_del_notebook_aportado(self):
        modelos = svm.modelos_originales()
        self.assertEqual(tuple(modelos), svm.NOMBRES)
        p = modelos["svm_lineal"].get_params()
        self.assertIsInstance(modelos["svm_lineal"], LinearSVC)
        self.assertEqual((p["C"], p["class_weight"], p["dual"], p["max_iter"]),
                         (0.010053940194693016, "balanced", "auto", 7000))
        self.assertEqual(modelos["regresion_logistica"].get_params()["C"], 1.0)
        self.assertEqual(modelos["arbol_decision"].get_params()["min_samples_leaf"], 1000)
        self.assertEqual(modelos["gradient_boosting"].get_params()["max_iter"], 75)
        self.assertTrue(all(m.get_params()["random_state"] == 42 for m in modelos.values()))

    def test_preparacion_coincide_con_el_escenario_original_sin_repetir_otros(self):
        e = self.experimento_sintetico()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = preparar_experimentos(e["particiones"], [])[svm.ESCENARIO[0]]
            with patch.object(svm, "preparar_experimentos", wraps=preparar_experimentos) as preparar:
                nuevo = svm.preparar_variables(e)
        self.assertEqual(preparar.call_args.kwargs["escenarios"], [svm.ESCENARIO])
        for conjunto in e["particiones"]:
            pd.testing.assert_frame_equal(nuevo["X"][conjunto], original["X"][conjunto])
            pd.testing.assert_series_equal(nuevo["y"][conjunto], original["y"][conjunto])

    def test_cargador_delega_una_sola_muestra_y_valida_los_tamanos(self):
        e = cargar_experimento_inicial(modo="consulta", raiz=self.RAIZ)
        muestra = MagicMock()
        muestra.__len__.return_value = 220003
        muestra.default.unique.return_value = np.array([0, 1])
        e["muestra"] = muestra
        with patch.object(svm, "cargar_experimento_inicial", return_value=e) as cargar:
            resultado = svm.cargar_muestra("fuente.csv", self.RAIZ)
        cargar.assert_called_once_with(ruta_datos="fuente.csv", modo="reproduccion", raiz=self.RAIZ, verbose=True)
        self.assertIs(resultado, e)
        muestra.__len__.return_value = 220000
        with patch.object(svm, "cargar_experimento_inicial", return_value=e), self.assertRaisesRegex(ValueError, "220,003"):
            svm.cargar_muestra()

    def test_margenes_no_son_probabilidades_y_umbral_no_usa_el_test(self):
        y = {n: np.array([0, 0, 1, 1]) for n in ("entrenamiento", "validacion", "prueba")}
        scores = {n: np.array([-10., -9., -8., -7.]) for n in y}
        with patch.object(svm, "elegir_umbral", wraps=svm.elegir_umbral) as elegir:
            filas = svm.metricas_modelo("svm_lineal", scores, y, .01)
        elegir.assert_called_once()
        self.assertIs(elegir.call_args.args[0], y["validacion"])
        self.assertIs(elegir.call_args.args[1], scores["validacion"])
        self.assertTrue(all(f["umbral_validacion"] == -8 for f in filas))
        self.assertTrue(all(f["ROC_AUC"] == roc_auc_score(y["validacion"], scores["validacion"]) for f in filas))
        self.assertTrue(all(f["AP"] == average_precision_score(y["validacion"], scores["validacion"]) for f in filas))
        self.assertFalse(any("brier" in k.lower() or "log_loss" in k.lower() for f in filas for k in f))
        y["prueba"] = 1 - y["prueba"]
        otras = svm.metricas_modelo("svm_lineal", scores, y, 99.)
        self.assertTrue(all(f["umbral_validacion"] == -8 for f in otras))

    def test_cuatro_ajustes_comparten_exactamente_las_matrices(self):
        datos = svm.preparar_variables(self.experimento_sintetico())
        modelos = svm.modelos_originales()
        with ExitStack() as stack:
            ajustes = {n: stack.enter_context(patch.object(m, "fit", wraps=m.fit)) for n, m in modelos.items()}
            stack.enter_context(patch.object(svm, "modelos_originales", return_value=modelos))
            metricas, _, scores, _ = svm.entrenar_modelos(datos)
        self.assertEqual(metricas.shape[0], 12)
        for n, ajustar in ajustes.items():
            ajustar.assert_called_once()
            self.assertIs(ajustar.call_args.args[0], datos["X"]["entrenamiento"])
            self.assertEqual(scores[n]["prueba"].shape[0], datos["X"]["prueba"].shape[0])

    def test_salidas_independientes_y_verificacion_sin_referencias_inventadas(self):
        e = self.experimento_sintetico()
        datos = svm.preparar_variables(e)
        y = {n: v.to_numpy() for n, v in datos["y"].items()}
        filas = []
        for n in svm.NOMBRES:
            scores = {c: np.arange(len(v), dtype=float) for c, v in y.items()}
            filas.extend(svm.metricas_modelo(n, scores, y, .1))
        metricas = pd.DataFrame(filas)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            r = Path(temp)
            sesion = SesionNotebook(r, "05_svm", True)
            svm.guardar_resultados(sesion.salida, e, datos, metricas)
            pares = [(r / "outputs/svm_muestra" / n, sesion.salida / n) for n in svm.ARCHIVOS_VERIFICABLES]
            verificacion = sesion.finalizar(pares, svm.configuracion_experimento())
            self.assertEqual(set(verificacion.Estado), {"No verificable"})
            archivo = sesion.salida / "metricas_svm_muestra.csv"
            original = archivo.read_bytes()
            with self.assertRaises(FileExistsError):
                svm.guardar_resultados(sesion.salida, e, datos, metricas)
            self.assertEqual(original, archivo.read_bytes())
            copia = r / "comparacion.csv"
            cambiados = metricas.copy()
            cambiados["segundos_entrenamiento"] = 99.
            cambiados.to_csv(copia, index=False)
            self.assertEqual(set(comparar_archivos(archivo, copia).Estado), {"Coincide"})
            cambiados.loc[0, "ROC_AUC"] += .01
            cambiados.to_csv(copia, index=False)
            self.assertIn("Difiere", set(comparar_archivos(archivo, copia).Estado))

    def test_reproduccion_se_detiene_antes_del_dataset_en_destino_nuevo(self):
        class FinPrueba(Exception):
            pass
        with patch("pathlib.Path.mkdir", autospec=True) as crear, \
                patch("pathlib.Path.write_text", autospec=True) as escribir, \
                patch.object(svm, "cargar_muestra", side_effect=FinPrueba) as cargar:
            with self.assertRaises(FinPrueba):
                self.ejecutar_notebook(self.RAIZ, reproducir=True)
        cargar.assert_called_once_with(None, self.RAIZ)
        crear.assert_called_once()
        destino = crear.call_args.args[0]
        self.assertTrue(destino.is_relative_to(self.RAIZ / "outputs/reproducciones"))
        self.assertTrue(destino.name.startswith("05_svm_"))
        self.assertTrue(all(c.args[0].is_relative_to(destino) for c in escribir.call_args_list))

    def test_consulta_con_referencia_temporal_no_ajusta_modelos(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz = Path(temp)
            (raiz / "src").mkdir()
            (raiz / "src/config.py").touch()
            historico = raiz / "outputs/svm_muestra"
            historico.mkdir(parents=True)
            e = self.experimento_sintetico()
            datos = svm.preparar_variables(e)
            metricas = pd.DataFrame([
                {"modelo": m, "conjunto": c, "ROC_AUC": .7, "AP": .4, "segundos_entrenamiento": .1}
                for m in svm.NOMBRES for c in ("entrenamiento", "validacion", "prueba")])
            svm.guardar_resultados(historico, e, datos, metricas)
            with ExitStack() as stack:
                for target in ["src.svm_muestra.cargar_muestra", "src.svm_muestra.entrenar_modelos",
                               "src.svm_muestra.preparar_variables", "pathlib.Path.mkdir",
                               "pathlib.Path.write_text", "pandas.DataFrame.to_csv"]:
                    stack.enter_context(patch(target, side_effect=AssertionError(target)))
                stack.enter_context(patch("IPython.display.display"))
                env = self.ejecutar_notebook(raiz)
            self.assertIsNone(env["modelos"])
            self.assertEqual(set(env["verificacion"].Estado), {"No verificado"})
            self.assertFalse((raiz / "outputs/reproducciones").exists())
            self.assertEqual(len(env["metricas"]), 12)


if __name__ == "__main__":
    unittest.main()
