"""Los mismos escenarios y reglas para el notebook y el script."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import features as ft
from . import evaluate as ev
from .config import SEMILLA, UMBRAL_FALTANTES, UMBRAL_REVISION

ESCENARIOS = [
    ("logistica_sin_indicadores", False, False),
    ("logistica_con_indicadores", False, True),
    ("logistica_con_precio_lc", True, True),
]


def preparar_experimentos(particiones, columnas_vacias):
    """Aprende la preparación solo con entrenamiento; mantiene filas e índices."""
    preparados = {}
    for nombre, precio, indicadores in ESCENARIOS:
        candidatas = ft.seleccionar_predictoras(
            particiones['entrenamiento'], columnas_vacias, incluir_precio_lc=precio)
        originales = {
            conjunto: ft.preparar_matriz(df[candidatas + ['fecha_prestamo']])
                .drop(columns='fecha_prestamo')
            for conjunto, df in particiones.items()
        }
        columnas, indicadoras, resumen = ft.filtrar_por_faltantes(
            originales['entrenamiento'], list(originales['entrenamiento'].columns),
            UMBRAL_FALTANTES, UMBRAL_REVISION)
        if not indicadores:
            indicadoras = []
            resumen.loc[resumen['decision'] == 'imputar + indicador', 'decision'] = 'imputar'
        preprocesador, _ = ft.construir_preprocesador(
            originales['entrenamiento'][columnas], indicadoras)
        preprocesador.fit(originales['entrenamiento'][columnas])
        nombres = preprocesador.get_feature_names_out()
        matrices = {}
        objetivos = {}
        for conjunto, original in originales.items():
            matriz = pd.DataFrame(preprocesador.transform(original[columnas]),
                                  index=original.index, columns=nombres)
            assert np.isfinite(matriz.to_numpy()).all()
            objetivos[conjunto] = particiones[conjunto]['default'].copy()
            assert matriz.index.equals(objetivos[conjunto].index)
            matrices[conjunto] = matriz
        preparados[nombre] = {
            'X': matrices, 'y': objetivos, 'preprocesador': preprocesador,
            'faltantes': resumen,
            'variables': {
                'predictoras_evaluadas': len(candidatas),
                'predictoras_finales': len(columnas),
                'columnas_modelo': len(nombres),
                'con_indicador_faltante': len(indicadoras),
                'excluidas_por_faltantes': int((resumen['decision'] == 'excluir').sum()),
            },
        }
    return preparados


def entrenar_experimentos(preparados):
    """Entrena tres logísticas; elige cada umbral únicamente en validación."""
    resultados, modelos, scores = {}, {}, {}
    for nombre, datos in preparados.items():
        X, y = datos['X'], datos['y']
        modelo = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0,
                                    random_state=SEMILLA)
        modelo.fit(X['entrenamiento'], y['entrenamiento'])
        probabilidades = {c: modelo.predict_proba(tabla)[:, 1] for c, tabla in X.items()}
        umbral = ev.elegir_umbral(y['validacion'], probabilidades['validacion'])
        resultados[nombre] = {c: ev.evaluar(y[c], score, umbral)
                              for c, score in probabilidades.items()}
        modelos[nombre] = modelo
        scores[nombre] = (y['validacion'].to_numpy(), probabilidades['validacion'])
    base = preparados['logistica_sin_indicadores']['y']
    tasa = float(base['entrenamiento'].mean())
    resultados['clase_mayoritaria'] = {
        c: ev.evaluar(y, np.full(len(y), tasa), 1.1) for c, y in base.items()
    }
    scores['clase_mayoritaria'] = (base['validacion'].to_numpy(),
                                   np.full(len(base['validacion']), tasa))
    return resultados, modelos, scores
