"""Controles de consulta y de la receta inicial con observaciones sintéticas."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import arbol, boosting, experimento_inicial as ei
from src.data import escanear_y_muestrear


class ExperimentoInicialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raiz = Path(__file__).resolve().parents[1]

    def test_consulta_no_localiza_ni_lee_el_dataset(self):
        with patch.object(ei, "localizar_datos", side_effect=AssertionError("CSV")), \
                patch.object(ei, "escanear_y_muestrear", side_effect=AssertionError("muestreo")):
            consulta = ei.cargar_experimento_inicial()
        self.assertIsNone(consulta["muestra"])
        self.assertIsNone(consulta["particiones"])
        self.assertEqual(consulta["resumen"]["filas"].tolist(), [73701, 109382, 36920])
        self.assertEqual(int(consulta["resumen"]["filas"].sum()), 220003)

    def test_carga_delega_una_vez_sin_cambiar_semilla_bloques_ni_cortes(self):
        muestra = pd.DataFrame({"anio": [2014, 2014, 2015, 2016, 2017, 2018],
                                "default": [0, 1, 0, 1, 0, 1]})
        audit = {"total_filas": 10, "faltantes_globales": pd.Series({"vacia": 10, "dato": 0})}
        with patch.object(ei, "localizar_datos", return_value=Path("sintetico.csv")), \
                patch.object(ei, "escanear_y_muestrear", return_value=(muestra, audit)) as scan:
            e = ei.cargar_experimento_inicial(tamano_muestra=6, modo="reproduccion")
        scan.assert_called_once_with(Path("sintetico.csv"), tamano_muestra=6,
                                     tamano_bloque=50000, semilla=42, verbose=False)
        self.assertEqual(e["columnas_vacias"], ["vacia"])
        self.assertEqual([len(e["particiones"][k]) for k in ei.CONJUNTOS], [2, 2, 2])
        self.assertIs(e["muestra"], muestra)
        self.assertIs(e["auditoria"], audit)

    def test_receta_sintetica_coincide_con_el_muestreador_original(self):
        n = 60
        raw = pd.DataFrame({
            "id": [str(i) for i in range(n)], "policy_code": [1] * n,
            "loan_status": ["Fully Paid", "Charged Off", "Current"] * 20,
            "issue_d": [f"Jan-{2010 + (i % 9)}" for i in range(n)],
            "earliest_cr_line": ["Jan-2000"] * n, "term": ["36 months"] * n,
            "loan_amnt": [1000.] * n, "annual_inc": [50000.] * n,
            "dti": [12.] * n, "revol_bal": [200.] * n,
            "fico_range_low": [680.] * n, "fico_range_high": [684.] * n,
        })
        with tempfile.TemporaryDirectory() as temp:
            archivo = Path(temp) / "mini.csv"
            raw.to_csv(archivo, index=False)
            esperado, _ = escanear_y_muestrear(archivo, tamano_muestra=30, verbose=False)
            actual = ei.cargar_experimento_inicial(archivo, 30, modo="reproduccion")
        pd.testing.assert_frame_equal(actual["muestra"], esperado)
        self.assertEqual(len(actual["muestra"]["id"].unique()), len(actual["muestra"]))
        self.assertEqual(set(actual["muestra"]["loan_status"]), {"Fully Paid", "Charged Off"})
        indices = [set(p.index) for p in actual["particiones"].values()]
        self.assertTrue(all(not indices[i] & indices[j] for i in range(3) for j in range(i)))

    def test_referencias_rechazan_otra_poblacion(self):
        e = ei.cargar_experimento_inicial()
        refs = ei.cargar_referencias()
        ei.validar_referencias(e, list(refs.values()))
        e["resumen"].loc["validacion", "filas"] += 1
        with self.assertRaisesRegex(ValueError, "Filas o etiquetas"):
            ei.validar_referencias(e, [refs["logistica"]])

    def test_predictoras_contrastadas_sin_entrenar_logistica(self):
        d = pd.read_csv(self.raiz / "outputs/decisiones_faltantes_logistica_sin_indicadores.csv", index_col=0)
        columnas = d.index[d["decision"] != "excluir"].tolist()
        ei.comprobar_predictoras({"columnas": columnas})
        with self.assertRaises(ValueError):
            ei.comprobar_predictoras({"columnas": columnas[:-1]})

    def test_notebook_boosting_no_depende_del_ejecutor_y_conserva_108_configuraciones(self):
        nb = json.loads((self.raiz / "notebooks/03_gradient_boosting.ipynb").read_text(encoding="utf-8"))
        codigo = "\n".join("".join(c["source"]) for c in nb["cells"]
                            if c["cell_type"] == "code")
        self.assertNotIn("src." + "run_boosting", codigo)
        self.assertIn('MODO_BUSQUEDA = "completo"', codigo)
        self.assertIn('"prueba"', codigo)
        self.assertIn("modelo.buscar_hiperparametros(datos, grilla=grilla)", codigo)
        self.assertEqual(int(np.prod([len(v) for v in boosting.GRILLA.values()])), 108)

    def test_preparaciones_preservan_nan_numerico_y_categoria_ausente(self):
        train = pd.DataFrame({"numero": [1., np.nan, 3.], "categoria": ["A", np.nan, "B"]})
        for construir in [arbol.construir_preprocesador_arbol,
                          boosting.construir_preprocesador_boosting]:
            pre = construir(train)
            matriz = pre.fit_transform(train)
            self.assertTrue(np.isnan(matriz[1, 0]))
            self.assertIn("categoria_Sin información", pre.get_feature_names_out())
            np.testing.assert_array_equal(matriz[[0, 2], 0], [1., 3.])

    def test_destinos_independientes_y_modo_consulta_protegidos(self):
        with tempfile.TemporaryDirectory() as temp:
            salida = Path(temp) / "salida"
            with self.assertRaises(ValueError):
                ei.rutas_reproduccion("prueba", salida, salida / "figuras")
    def test_notebooks_en_consulta_no_entrenan_ni_muestrean(self):
        targets = [
            "src.experimento_inicial.escanear_y_muestrear",
            "src.experimento_inicial.localizar_datos",
            "src.arbol.preparar_datos", "src.boosting.preparar_datos",
            "src.arbol.buscar_hiperparametros", "src.boosting.buscar_hiperparametros",
            "src.experimentos.entrenar_experimentos",
            "src.arbol.DecisionTreeClassifier.fit",
            "src.boosting.HistGradientBoostingClassifier.fit",
            "src.experimentos.LogisticRegression.fit",
        ]
        for cwd in [self.raiz, self.raiz / "notebooks"]:
            for name in ["02_arbol_decision.ipynb", "03_gradient_boosting.ipynb"]:
                nb = json.loads((self.raiz / "notebooks" / name).read_text(encoding="utf-8"))
                env = {"__name__": "__notebook_consulta__"}
                with ExitStack() as stack:
                    for target in targets:
                        stack.enter_context(patch(target, side_effect=AssertionError(target)))
                    stack.enter_context(patch("pathlib.Path.cwd", return_value=cwd))
                    stack.enter_context(patch("IPython.display.display"))
                    stack.enter_context(patch("IPython.display.Image"))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    for cell in nb["cells"]:
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), name, "exec"), env)
                self.assertFalse(env["REPRODUCIR"])
                self.assertIsNone(env["experimento"]["particiones"])


if __name__ == "__main__":
    unittest.main()
