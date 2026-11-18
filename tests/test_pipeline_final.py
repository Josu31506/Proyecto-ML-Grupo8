"""Pruebas del flujo congelado exclusivamente con préstamos sintéticos."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from src import etapas_finales as ef, features as ft
from src import run_final as rf


class PipelineFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_base, n_cal, n_test = 80, 40, 20
        n = n_base + n_cal + n_test
        cls.raw = pd.DataFrame({
            "id": [str(i) for i in range(n)],
            "issue_d": ["Jan-2015"] * n_base + ["Jan-2016"] * n_cal + ["Jan-2017"] * n_test,
            "loan_status": ["Fully Paid" if i % 2 == 0 else "Charged Off" for i in range(n)],
            "loan_amnt": np.arange(n, dtype=float) * 7 + (np.arange(n) % 2) * 3000 + 1000,
            "annual_inc": np.r_[np.repeat(50000., n_base), np.repeat(50000., n_cal + n_test)],
            "open_acc": [np.nan] * 61 + [2.] * (n - 61),
            "home_ownership": ["OWN" if i % 3 else "RENT" for i in range(n)],
            "term": ["36 months"] * n,
            "earliest_cr_line": ["Jan-2000"] * n,
            "total_pymnt": [900.] * n,
            "hardship_flag": ["N"] * n,
        })
        cls.raw.loc[:9, "annual_inc"] = np.nan
        cls.candidatas = ft.seleccionar_predictoras(cls.raw, [], incluir_precio_lc=False)
        cls.preparados = ef._preparar_bloque(cls.raw.copy())
        cls.dev = cls.preparados[cls.preparados.anio <= 2016].copy()
        cls.test = cls.preparados[cls.preparados.anio == 2017].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.modelos = {m.nombre: m for m in ef.entrenar_pipelines_finales(cls.dev, cls.candidatas)}
        cls.raw_test = cls.raw.iloc[-n_test:].drop(columns=["loan_status", "id", "total_pymnt", "hardship_flag", "open_acc"]).copy()

    def test_particiones_temporales(self):
        p = ef.separar_periodos_finales(self.preparados)
        self.assertEqual([int(p[k].sum()) for k in p], [80, 40, 20])
        self.assertFalse((p["entrenamiento"] & p["calibracion"]).any())
        self.assertFalse((p["test"] & p["calibracion"]).any())

    def test_leakage_excluido(self):
        for c in ["loan_status", "issue_d", "id", "total_pymnt", "hardship_flag"]:
            self.assertNotIn(c, self.candidatas)
        for modelo in self.modelos.values():
            self.assertNotIn("total_pymnt", modelo.variables)
            self.assertNotIn("default", modelo.columnas_entrada)
            self.assertNotIn("loan_status", modelo.columnas_entrada)

    def test_selector_y_mediana_solo_base(self):
        mediana = self.dev.loc[self.dev.anio <= 2015, "annual_inc"].median()
        for modelo in self.modelos.values():
            self.assertNotIn("open_acc", modelo.variables)
            self.assertAlmostEqual(modelo.metadatos["faltantes_base_pct"]["open_acc"], 76.25)
            num = modelo.preprocesador.named_transformers_["num"]
            columnas = next(cols for nombre, _, cols in modelo.preprocesador.transformers_ if nombre == "num")
            indice = columnas.index("annual_inc")
            self.assertEqual(num.named_steps["imputar"].statistics_[indice], mediana)
            self.assertEqual(num.named_steps["escalar"].mean_[indice], 50000)
        modificado = self.dev.copy()
        modificado.loc[modificado.anio == 2016, "annual_inc"] = 1e9
        posterior = next(ef.entrenar_pipelines_finales(modificado, self.candidatas, modelos=["regresion_logistica"]))
        np.testing.assert_array_equal(posterior.preprocesador.named_transformers_["num"].named_steps["imputar"].statistics_,
            self.modelos["regresion_logistica"].preprocesador.named_transformers_["num"].named_steps["imputar"].statistics_)
        np.testing.assert_array_equal(posterior.preprocesador.named_transformers_["num"].named_steps["escalar"].mean_,
            self.modelos["regresion_logistica"].preprocesador.named_transformers_["num"].named_steps["escalar"].mean_)

    def test_consistencia_preparado_e_inferencia_sin_objetivo(self):
        X = ef.preparar_X(self.test, self.candidatas)
        for modelo in self.modelos.values():
            esperado = modelo.predecir_preparado(X)
            actual = modelo.predict_proba(self.raw_test)[:, 1]
            np.testing.assert_allclose(actual, esperado, atol=0, rtol=0)
            self.assertGreater(np.ptp(actual), 1e-4)
            np.testing.assert_array_equal(modelo.preparar_entrada(self.raw_test)[modelo.variables].to_numpy(), X[modelo.variables].to_numpy())
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_manejo_de_columna_ausente_y_categoria_nueva(self):
        for modelo in self.modelos.values():
            entrada = self.raw_test.copy()
            with self.assertRaisesRegex(ValueError, "Faltan columnas"):
                modelo.predict_proba(entrada.drop(columns="annual_inc"))
            entrada["home_ownership"] = "CATEGORIA_NUEVA"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = modelo.predict_proba(entrada)
            self.assertEqual(p.shape, (20, 2))
            np.testing.assert_allclose(p.sum(axis=1), 1)
            entrada["home_ownership"] = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertTrue(np.isfinite(modelo.predict_proba(entrada)).all())

    def test_serializacion_y_no_sobrescritura(self):
        modelo = self.modelos["gradient_boosting"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "modelo.joblib"
            rf.guardar_artefacto(modelo, path)
            loaded = rf.cargar_artefacto(path)
            np.testing.assert_array_equal(modelo.predict_proba(self.raw_test), loaded.predict_proba(self.raw_test))
            with self.assertRaises(FileExistsError):
                rf.guardar_artefacto(modelo, path)
            with self.assertRaises(FileExistsError):
                rf.carpeta_nueva(td)

    def test_semilla_reproducible(self):
        repetido = next(ef.entrenar_pipelines_finales(self.dev, self.candidatas, modelos=["regresion_logistica"]))
        np.testing.assert_array_equal(repetido.predict_proba(self.raw_test), self.modelos["regresion_logistica"].predict_proba(self.raw_test))

    def test_equivalencia_con_evaluacion_historica(self):
        original, pred = ef.entrenar_final_y_evaluar(self.dev, self.candidatas, self.test,
                                                  {m: "Sigmoide" for m in self.modelos})
        for nombre, modelo in self.modelos.items():
            actual = modelo.predict_proba(self.raw_test)[:, 1]
            np.testing.assert_allclose(actual, pred[nombre], atol=1e-12, rtol=0)
            metricas = ef.calcular_metricas(self.test.default, actual)
            row = original[original.modelo == nombre].iloc[0]
            for k in ["ROC_AUC", "PR_AUC", "Brier_Score", "Log_Loss"]:
                self.assertAlmostEqual(metricas[k], row[k], places=12)

    def test_sigmoide_equivalente(self):
        s = np.linspace(.05, .95, 40)
        y = np.arange(40) % 2
        nuevo = ef.ajustar_sigmoide(s, y, "probabilidad")
        anterior, iso = ef.ajustar_calibradores(s, y, "probabilidad")
        a, _ = ef.predecir_calibrado(anterior, iso, s, "probabilidad")
        b, _ = ef.predecir_calibrado(nuevo, iso, s, "probabilidad")
        np.testing.assert_array_equal(a, b)

    def test_bloqueo_test_previo_a_cualquier_carga(self):
        with patch.object(rf, "cargar_modelos") as cargar:
            with self.assertRaisesRegex(ValueError, "confirmar-test"):
                rf.evaluar("no_existe", "no_existe.csv", "test", confirmar_test=False)
            cargar.assert_not_called()
        rf.validar_evaluacion("test", True)

    def test_configuracion_historica_y_no_test_en_entrenar(self):
        self.assertEqual(rf.verificar_configuracion()["umbral_faltantes_train_pct"], 50)
        with tempfile.TemporaryDirectory() as td:
            destino = Path(td) / "modelos"
            with patch.object(ef, "cargar_desarrollo", return_value=(self.dev, self.candidatas, Path("sintetico.csv"))) as loader,                  patch.object(ef, "entrenar_pipelines_finales", return_value=iter(self.modelos.values())),                  patch.object(ef, "auditar_y_cargar_test") as test_loader:
                rf.entrenar("sintetico.csv", destino)
                self.assertEqual(loader.call_args.kwargs["ultimo_anio"], 2016)
                test_loader.assert_not_called()
            manifest = json.loads((destino / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["test_evaluado_en_esta_ejecucion"])
            self.assertEqual(len(rf.cargar_modelos(destino)), 3)
            archivo = Path(td) / "nuevos.csv"
            self.raw_test.to_csv(archivo, index=False)
            salida = Path(td) / "predicciones.csv"
            rf.predecir(destino, archivo, salida)
            self.assertEqual(len(pd.read_csv(salida)), 20)
            with self.assertRaises(FileExistsError):
                rf.predecir(destino, archivo, salida)

    def test_evaluacion_explicita_sintetica_sin_reentrenar(self):
        with tempfile.TemporaryDirectory() as td:
            ruta = Path(td) / "sintetico.csv"
            self.raw.to_csv(ruta, index=False)
            with patch.object(rf, "cargar_modelos", return_value=self.modelos), \
                 patch.object(ef, "entrenar_pipelines_finales", side_effect=AssertionError("No debe entrenar")):
                validacion = rf.evaluar("sinteticos", ruta, "validacion", salida=Path(td) / "validacion")
                alcance = json.loads((validacion / "alcance.json").read_text(encoding="utf-8"))
                self.assertIn("no independiente", alcance["interpretacion"])
                self.assertTrue((validacion / "evaluacion.png").exists())
                test = rf.evaluar("sinteticos", ruta, "test", confirmar_test=True, salida=Path(td) / "test")
                metricas = pd.read_csv(test / "metricas.csv")
                self.assertEqual(metricas.n.tolist(), [20, 20, 20])
                self.assertTrue((test / "errores_segmentos.csv").exists())

    def test_version_incompatible(self):
        copia = copy.deepcopy(self.modelos["gradient_boosting"])
        copia.metadatos["versiones"]["scikit-learn"] = "version_distinta"
        with tempfile.TemporaryDirectory() as td:
            archivo = Path(td) / "modelo.joblib"
            rf.guardar_artefacto(copia, archivo)
            with self.assertRaisesRegex(ValueError, "Versión incompatible"):
                rf.cargar_artefacto(archivo)


if __name__ == "__main__":
    unittest.main()
