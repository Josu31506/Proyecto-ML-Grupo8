import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.avance import cargar_avance, preparar_avance, comparar_modelos


class AvanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ruta = Path(self.tmp.name) / 'prestamos.csv'
        filas = []
        for anio in [2013, 2015, 2017]:
            for i in range(20):
                filas.append({
                    'issue_d': f'Jan-{anio}', 'earliest_cr_line': 'Jan-2000',
                    'loan_status': ['Fully Paid', 'Charged Off', 'Current', 'Fully Paid'][i % 4],
                    'term': ' 36 months', 'loan_amnt': 1000 + i * 50,
                    'annual_inc': np.nan if i == 0 else 20000 + i * 100,
                    'home_ownership': 'RENT' if i % 2 else 'OWN',
                    'solo_reciente': np.nan if anio == 2013 else i,
                    'total_pymnt': i * 100, 'recoveries': i,
                })
        pd.DataFrame(filas).to_csv(self.ruta, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_completo_conserva_solo_elegibles_y_excluye_futuro(self):
        datos, auditoria, cobertura = cargar_avance(self.ruta, None, 13)
        self.assertEqual(len(datos), 45)
        self.assertEqual(auditoria['filas_fuente'], 60)
        self.assertEqual(auditoria['fraccion'], 1)
        self.assertNotIn('recoveries', datos)
        self.assertNotIn('total_pymnt', datos)
        self.assertNotIn('solo_reciente', datos)
        self.assertEqual(cobertura['total_con_estado'].sum(), 60)
        self.assertEqual(cobertura['elegibles'].sum(), 45)

    def test_muestra_reproducible(self):
        a, _, _ = cargar_avance(self.ruta, 20, 13)
        b, _, _ = cargar_avance(self.ruta, 20, 13)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a['loan_status'].isin(['Fully Paid', 'Charged Off']).all())

    def test_validacion_no_modifica_preparacion_y_prueba_no_se_evalua(self):
        datos, _, _ = cargar_avance(self.ruta, None, 13)
        X, V, y, z, _ = preparar_avance(datos)
        alterado = datos.copy()
        alterado.loc[alterado['anio'] >= 2015, 'annual_inc'] = 999999999
        X2, _, _, _, _ = preparar_avance(alterado)
        np.testing.assert_array_equal(X, X2)
        resultados, _ = comparar_modelos(X, V, y, z)
        self.assertEqual(len(y), 15)
        self.assertEqual(len(z), 15)
        for conjuntos in resultados.values():
            self.assertEqual(set(conjuntos), {'entrenamiento', 'validacion'})

    def test_fecha_invalida_no_se_oculta(self):
        datos = pd.read_csv(self.ruta)
        datos.loc[0, 'issue_d'] = 'fecha incorrecta'
        datos.to_csv(self.ruta, index=False)
        with self.assertRaises(ValueError):
            cargar_avance(self.ruta, None, 13)


if __name__ == '__main__':
    unittest.main()
