"""Verifica controles de consulta sin ajustar modelos ni evaluar Lending Club."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import IPython.display
import matplotlib.pyplot as plt
import pandas as pd
from src import etapas_finales, modelado


class NotebookReproduccionTests(unittest.TestCase):
    RAIZ = Path(__file__).resolve().parents[1]
    ARCHIVOS = ("09_optimizacion_final.ipynb", "10_calibracion.ipynb", "11_evaluacion_final.ipynb")

    def ejecutar(self, archivo, cwd=None, reproducir=False, stack=None):
        nb = json.loads((self.RAIZ / "notebooks" / archivo).read_text(encoding="utf-8"))
        env = {"__name__": "__notebook_consulta__"}
        with patch("pathlib.Path.cwd", return_value=cwd or self.RAIZ), redirect_stdout(io.StringIO()):
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                code = "".join(cell["source"])
                if reproducir:
                    code = code.replace("REPRODUCIR = False", "REPRODUCIR = True")
                exec(compile(code, archivo, "exec"), env)
        return env

    def bloquear_operaciones(self, stack):
        targets = [
            "src.config.localizar_datos",
            "src.etapas_finales.cargar_desarrollo",
            "src.etapas_finales.auditar_y_cargar_test",
            "src.etapas_finales.correr_corte",
            "src.etapas_finales.entrenar_final_y_evaluar",
            "src.analisis_errores.ejecutar_analisis",
            "sklearn.linear_model.LogisticRegression.fit",
            "sklearn.svm.LinearSVC.fit",
            "sklearn.ensemble.HistGradientBoostingClassifier.fit",
            "sklearn.pipeline.Pipeline.fit",
            "sklearn.compose.ColumnTransformer.fit",
            "sklearn.metrics.roc_auc_score",
            "sklearn.metrics.average_precision_score",
            "sklearn.metrics.brier_score_loss",
            "sklearn.metrics.log_loss",
            "sklearn.calibration.calibration_curve",
            "pandas.DataFrame.to_csv",
            "pathlib.Path.write_text",
            "pathlib.Path.mkdir",
            "matplotlib.figure.Figure.savefig",
            "matplotlib.pyplot.subplots",
        ]
        for target in targets:
            stack.enter_context(patch(target, side_effect=AssertionError(target)))
        stack.enter_context(patch("IPython.display.display"))
        stack.enter_context(patch("IPython.display.Image"))
        read_csv = pd.read_csv

        def leer_agregados(path, *args, **kwargs):
            name = Path(path).name
            if name.startswith(("accepted_", "rejected_", "predicciones")):
                raise AssertionError("Consulta intentó leer observaciones o predicciones individuales.")
            return read_csv(path, *args, **kwargs)

        stack.enter_context(patch("pandas.read_csv", side_effect=leer_agregados))

    def test_consulta_de_09_10_11_desde_raiz_y_notebooks_sin_entrenar_ni_escribir(self):
        for archivo in self.ARCHIVOS:
            for cwd in (self.RAIZ, self.RAIZ / "notebooks"):
                with self.subTest(archivo=archivo, cwd=cwd), ExitStack() as stack:
                    self.bloquear_operaciones(stack)
                    env = self.ejecutar(archivo, cwd)
                    self.assertFalse(env["REPRODUCIR"])
                    self.assertEqual(env["SALIDA"], env["ORIGEN"])

    def test_cambio_de_modo_sin_reinicializar_no_permite_destinos_historicos(self):
        for archivo in self.ARCHIVOS:
            with self.subTest(archivo=archivo), ExitStack() as stack:
                self.bloquear_operaciones(stack)
                env = self.ejecutar(archivo)
                env["REPRODUCIR"] = True
                with self.assertRaisesRegex(RuntimeError, "Reinicia el kernel"):
                    env["exigir_reproduccion"]()

    def test_reproduccion_prepara_destino_nuevo_y_se_detiene_antes_del_dataset(self):
        class DatasetBloqueado(Exception):
            pass

        loaders = ["src.config.localizar_datos", "src.etapas_finales.cargar_desarrollo", "src.etapas_finales.auditar_y_cargar_test"]
        for archivo, loader in zip(self.ARCHIVOS, loaders):
            with self.subTest(archivo=archivo), ExitStack() as stack:
                mkdir = stack.enter_context(patch("pathlib.Path.mkdir", autospec=True))
                escribir = stack.enter_context(patch("pathlib.Path.write_text", autospec=True))
                cargar = stack.enter_context(patch(loader, side_effect=DatasetBloqueado))
                stack.enter_context(patch("IPython.display.display"))
                stack.enter_context(patch("pandas.DataFrame.to_csv", side_effect=AssertionError("CSV")))
                with self.assertRaises(DatasetBloqueado):
                    self.ejecutar(archivo, reproducir=True)
                cargar.assert_called_once()
                self.assertEqual(mkdir.call_count, 1)
                destino = mkdir.call_args.args[0]
                self.assertTrue(destino.is_relative_to(self.RAIZ / "outputs/reproducciones"))
                self.assertFalse(mkdir.call_args.kwargs["exist_ok"])
                for call in escribir.call_args_list:
                    self.assertTrue(call.args[0].is_relative_to(destino))


if __name__ == "__main__":
    unittest.main()
