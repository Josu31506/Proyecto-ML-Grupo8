"""Pruebas ligeras de consistencia para la entrega del proyecto."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import features
from src.analisis_errores import cargar_segmentos_test, tabla_metricas_globales
from src.etapas_finales import _preparar_bloque
from src.modelado import crear_cortes_temporales


class IntegracionTests(unittest.TestCase):
    def test_objetivo_y_exclusion_de_informacion_posterior(self):
        df = pd.DataFrame({
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "issue_d": ["Jan-2015"] * 3,
            "loan_amnt": [1000, 2000, 3000],
            "total_pymnt": [1000, 500, 0],
            "recoveries": [0, 300, 0],
            "hardship_status": [None] * 3,
        })
        preparada = _preparar_bloque(df.copy())
        self.assertEqual(preparada["default"].iloc[:2].tolist(), [0, 1])
        predictoras = features.seleccionar_predictoras(preparada, [])
        self.assertIn("loan_amnt", predictoras)
        self.assertNotIn("loan_status", predictoras)
        self.assertNotIn("default", predictoras)
        self.assertNotIn("total_pymnt", predictoras)
        self.assertNotIn("recoveries", predictoras)
        self.assertNotIn("hardship_status", predictoras)

    def test_seleccion_por_faltantes_es_estricta_y_solo_usa_referencia(self):
        x = pd.DataFrame({"poca_falta": [1.0, np.nan, 3.0, 4.0],
                          "diez_pct": [1.0, np.nan, np.nan, np.nan],
                          "constante_vacia": [np.nan] * 4})
        dentro, fuera, tabla = features.seleccionar_por_cobertura(
            x, list(x.columns), umbral_excluir=50
        )
        self.assertEqual(dentro, ["poca_falta"])
        self.assertEqual(set(fuera), {"diez_pct", "constante_vacia"})
        self.assertEqual(tabla.loc["diez_pct", "porcentaje_faltante"], 75)

    def test_cortes_temporales_expansivos_son_disjuntos_por_anio(self):
        anios = [2012, 2012] + [anio for anio in range(2013, 2017) for _ in range(2)] + [2017]
        etiquetas = [0, 1] + [etiqueta for _ in range(2013, 2017) for etiqueta in (0, 1)] + [1]
        datos = pd.DataFrame({"anio": anios, "default": etiquetas})
        cortes = crear_cortes_temporales(datos)
        validacion = cortes[cortes["fold"].str.startswith("Validación")]
        self.assertEqual(validacion["anio_validacion"].tolist(), [2013, 2014, 2015, 2016])
        self.assertEqual(validacion["anio_final_entrenamiento"].tolist(), [2012, 2013, 2014, 2015])
        self.assertEqual(cortes.iloc[-1]["fold"], "Prueba reservada")

    def test_preprocesador_ajusta_train_y_tolera_categoria_nueva(self):
        train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "categoria": ["A", "B", "A"]})
        valid = pd.DataFrame({"n": [np.nan], "categoria": ["C"]})
        prep, _ = features.construir_preprocesador(train, con_indicador=[])
        x_train = prep.fit_transform(train)
        with self.assertWarns(UserWarning):
            x_valid = prep.transform(valid)
        self.assertEqual(x_train.shape[1], x_valid.shape[1])
        self.assertTrue(np.isfinite(x_valid).all())

    def test_analisis_de_errores_alinea_por_ano_y_etiqueta(self):
        pred = pd.DataFrame({"anio": [2017, 2017], "default": [0, 1],
                             "gradient_boosting": [0.2, 0.7]})
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "mini.csv"
            pd.DataFrame({
                "issue_d": ["Jan-2017", "Feb-2017", "Mar-2017"],
                "loan_status": ["Fully Paid", "Charged Off", "Current"],
                "term": [" 36 months", " 60 months", " 36 months"],
                "annual_inc": [30000, 80000, 40000],
                "application_type": ["Individual", "Joint App", "Individual"],
            }).to_csv(source, index=False)
            merged = cargar_segmentos_test(pred, ruta=source, chunksize=1)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged["plazo"].tolist(), ["36", "60"])
        m = tabla_metricas_globales(merged, modelos=["gradient_boosting"])
        self.assertEqual(int(m.iloc[0]["verdaderos_negativos"]), 1)
        self.assertEqual(int(m.iloc[0]["verdaderos_positivos"]), 1)

    def test_notebooks_tienen_json_valido_y_no_guardan_errores(self):
        raiz = Path(__file__).resolve().parents[1]
        notebooks = sorted((raiz / "notebooks").glob("*.ipynb"))
        self.assertGreaterEqual(len(notebooks), 10)
        for ruta in notebooks:
            doc = json.loads(ruta.read_text(encoding="utf-8"))
            self.assertEqual(doc.get("nbformat"), 4, ruta.name)
            for celda in doc.get("cells", []):
                self.assertIn(celda.get("cell_type"), {"code", "markdown"}, ruta.name)
                if celda.get("cell_type") == "code":
                    self.assertFalse(any(salida.get("output_type") == "error"
                                         for salida in celda.get("outputs", [])), ruta.name)


if __name__ == "__main__":
    unittest.main()
