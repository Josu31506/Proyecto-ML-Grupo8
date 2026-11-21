"""Pruebas sintéticas de los gráficos de Gradient Boosting."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import visualizacion_boosting as visual


class VisualizacionBoostingTests(unittest.TestCase):
    def test_genera_las_tres_figuras_en_el_destino_indicado(self):
        busqueda = pd.DataFrame({
            "max_leaf_nodes": [15, 31, 63, 15],
            "learning_rate": [.03, .05, .1, .1],
            "auc_validacion": [.68, .70, .69, .71],
            "brecha": [.03, .02, .04, .01],
        })
        importancia = pd.DataFrame({
            "variable": ["fico", "ingreso"],
            "caida_auc_media": [.02, .01],
            "caida_auc_desv": [.005, .003],
        })
        y = np.array([0, 0, 1, 1])
        puntuacion = np.array([.1, .4, .6, .9])

        with tempfile.TemporaryDirectory() as temporal:
            salida = Path(temporal) / "figuras"
            rutas = [
                visual.fig_busqueda(busqueda, .70, salida),
                visual.fig_importancia(importancia, salida),
                visual.fig_roc_comparacion(
                    [("Gradient Boosting", y, puntuacion)], salida,
                    referencias_auc={"Regresión logística": .71, "Árbol": .67}),
            ]
            self.assertEqual([p.name for p in rutas], [
                "14_boosting_busqueda.png", "15_boosting_importancia.png",
                "16_boosting_comparacion_roc.png",
            ])
            self.assertTrue(all(p.is_file() and p.stat().st_size > 0 for p in rutas))


if __name__ == "__main__":
    unittest.main()
