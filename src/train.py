"""Constructores auxiliares. Los escenarios oficiales se ejecutan en experimentos.py."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import SEMILLA


def construir_modelo(preprocesador, tipo="logistica"):
    """Une preprocesamiento y estimador en un unico Pipeline reproducible."""
    if tipo == "logistica":
        estimador = LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=SEMILLA)
    elif tipo == "clase_mayoritaria":
        estimador = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {tipo}")
    return Pipeline([("preprocesar", preprocesador), ("modelo", estimador)])


def coeficientes(modelo_ajustado, top=20):
    """Coeficientes de la regresion logistica, ordenados por magnitud.

    Como las numericas estan estandarizadas, el coeficiente se lee como el
    cambio en log-odds por una desviacion estandar de la variable.
    """
    import numpy as np
    import pandas as pd

    preprocesador = modelo_ajustado.named_steps["preprocesar"]
    estimador = modelo_ajustado.named_steps["modelo"]
    tabla = pd.DataFrame({
        "variable": preprocesador.get_feature_names_out(),
        "coeficiente": estimador.coef_[0],
    })
    tabla["odds_ratio"] = np.exp(tabla["coeficiente"])
    tabla["magnitud"] = tabla["coeficiente"].abs()
    return tabla.sort_values("magnitud", ascending=False).head(top)
