"""Transformaciones compartidas por los experimentos temporales."""
import numpy as np
import pandas as pd

def preparar_X(tabla, fecha_otorgamiento):
    X = tabla.copy()
    for col in ['earliest_cr_line', 'sec_app_earliest_cr_line']:
        if col in X:
            fecha_inicio = X[col]
            nombre = 'meses_desde_' + col
            X[nombre] = ((fecha_otorgamiento.dt.year - fecha_inicio.dt.year) * 12 + (fecha_otorgamiento.dt.month - fecha_inicio.dt.month)).astype('float64')
            X = X.drop(columns=col)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64').replace([np.inf, -np.inf], np.nan)
        else:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X
