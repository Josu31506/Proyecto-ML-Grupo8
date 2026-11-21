"""Pruebas sintéticas de los gráficos del experimento del árbol."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src import visualizacion_arbol as visual


class VisualizacionArbolTests(unittest.TestCase):
    def test_genera_las_cuatro_figuras_en_el_destino_indicado(self):
        grilla = {
            "max_depth": [2, 3, None],
            "min_samples_leaf": [1, 2],
            "class_weight": [None],
        }
        filas = []
        for profundidad, hoja, peso in itertools.product(*grilla.values()):
            filas.append({
                "max_depth": profundidad,
                "min_samples_leaf": hoja,
                "class_weight": peso,
                "auc_entrenamiento": .72 + hoja / 100,
                "auc_validacion": .68 + (profundidad or 1) / 100,
            })
        busqueda = pd.DataFrame(filas)
        seleccion = {"max_depth": 2, "min_samples_leaf": 1, "class_weight": None}
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1, 1])
        arbol = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
        importancia = pd.DataFrame({
            "variable": ["x"], "caida_auc_media": [.02], "caida_auc_desv": [.01]
        })
        curvas = [("Árbol", y, arbol.predict_proba(X)[:, 1])]

        with tempfile.TemporaryDirectory() as temporal:
            salida = Path(temporal) / "figuras"
            rutas = [
                visual.fig_complejidad(busqueda, seleccion, .71, grilla, salida),
                visual.fig_reglas(arbol, ["x"], salida),
                visual.fig_importancia(importancia, salida),
                visual.fig_roc(curvas, salida, {"Logística": .70}),
            ]
            self.assertEqual([p.name for p in rutas], [
                "10_arbol_complejidad.png", "11_arbol_reglas.png",
                "12_arbol_importancia.png", "13_comparacion_roc.png",
            ])
            self.assertTrue(all(p.is_file() and p.stat().st_size > 0 for p in rutas))


if __name__ == "__main__":
    unittest.main()
