"""Avance del proyecto: cobertura de datos, nuevos hallazgos y otro modelo.

python -m src.avance --modo muestra
python -m src.avance --modo completo
"""
import argparse
import gc
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from . import features as ft
from . import evaluate as ev
from .config import (SEMILLA, TAMANO_BLOQUE, TAMANO_MUESTRA, RAIZ,
                     ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA, localizar_datos,
                     UMBRAL_FALTANTES, UMBRAL_REVISION)
from .data import inferir_tipos, _normalizar


def cargar_avance(ruta, tamano_muestra=TAMANO_MUESTRA, tamano_bloque=TAMANO_BLOQUE):
    """Dos pasadas por bloques; solo carga las predictoras utilizables.

    None selecciona todos los elegibles. La primera pasada calcula cobertura
    y faltantes de entrenamiento; la segunda carga las filas seleccionadas.
    """
    if tamano_muestra is not None and tamano_muestra <= 0:
        raise ValueError('El tamaño de muestra debe ser positivo o None.')
    tipos, columnas = inferir_tipos(ruta)
    candidatas = ft.seleccionar_predictoras(pd.DataFrame(columns=columnas), [], False)
    lectura = [c for c in columnas if c in candidatas + ['loan_status', 'issue_d']]
    faltantes = pd.Series(0, index=candidatas, dtype='int64')
    total, elegibles_total, entrenamiento_total = 0, 0, 0
    estados = pd.Series(dtype='int64')
    cobertura = []
    with pd.read_csv(ruta, usecols=lectura, dtype={c: tipos[c] for c in lectura},
                              chunksize=tamano_bloque, skipinitialspace=True) as lector:
        for bloque in lector:
            total += len(bloque)
            estados = estados.add(bloque['loan_status'].fillna('SIN ESTADO').value_counts(), fill_value=0)
            # No ocultar fechas de otorgamiento mal formadas de préstamos con estado.
            con_estado = bloque['loan_status'].notna()
            pd.to_datetime(bloque.loc[con_estado, 'issue_d'], format='%b-%Y', errors='raise')
            bloque = _normalizar(bloque)
            if bloque.loc[con_estado, 'issue_d'].isna().any():
                raise ValueError('Hay préstamos con estado pero sin fecha de otorgamiento.')
            elegible = bloque['loan_status'].isin(['Fully Paid', 'Charged Off'])
            elegibles_total += int(elegible.sum())
            train = elegible & (bloque['issue_d'].dt.year < ANIO_CORTE_VALIDACION)
            entrenamiento_total += int(train.sum())
            faltantes += bloque.loc[train, candidatas].isna().sum()
            tabla = pd.DataFrame({
                'anio': bloque.loc[con_estado, 'issue_d'].dt.year,
                'plazo': bloque.loc[con_estado, 'term'],
                'estado': bloque.loc[con_estado, 'loan_status'],
            })
            cobertura.append(tabla.groupby(['anio', 'plazo', 'estado'], dropna=False).size())
    if entrenamiento_total == 0 or elegibles_total == 0:
        raise ValueError('No hay suficientes préstamos elegibles en entrenamiento.')
    porcentaje = faltantes / entrenamiento_total * 100
    # La exclusión en modo muestra se decide después, solo con SU entrenamiento.
    # Para modo completo se evita cargar las columnas que ya sabemos que se excluyen.
    conservadas = candidatas if tamano_muestra is not None else porcentaje[porcentaje <= UMBRAL_FALTANTES].index.tolist()
    lectura = [c for c in columnas if c in conservadas + ['loan_status', 'issue_d']]
    fraccion = 1.0 if tamano_muestra is None else min(1.0, tamano_muestra / elegibles_total)
    partes = []
    with pd.read_csv(
            ruta, usecols=lectura, dtype={c: tipos[c] for c in lectura},
            chunksize=tamano_bloque, skipinitialspace=True) as lector:
        for numero, bloque in enumerate(lector):
            bloque = _normalizar(bloque)
            elegibles = bloque.loc[bloque['loan_status'].isin(['Fully Paid', 'Charged Off'])]
            parte = elegibles if fraccion == 1 else elegibles.sample(frac=fraccion, random_state=SEMILLA + numero)
            partes.append(parte)
    datos = pd.concat(partes, ignore_index=True).copy()
    datos['default'] = datos['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    datos['fecha_prestamo'] = datos['issue_d']
    datos['anio'] = datos['issue_d'].dt.year
    assert datos['anio'].notna().all() and datos['default'].notna().all()
    if tamano_muestra is None:
        assert len(datos) == elegibles_total
    cobertura = pd.concat(cobertura).groupby(level=[0, 1, 2], dropna=False).sum().unstack('estado', fill_value=0)
    cobertura['total_con_estado'] = cobertura.sum(axis=1)
    cobertura['elegibles'] = cobertura.reindex(columns=['Fully Paid', 'Charged Off'], fill_value=0).sum(axis=1)
    cobertura['porcentaje_elegibles'] = 100 * cobertura['elegibles'] / cobertura['total_con_estado']
    cobertura['porcentaje_incumplimiento_elegibles'] = 100 * cobertura.get('Charged Off', 0) / cobertura['elegibles'].replace(0, np.nan)
    auditoria = {
        'filas_fuente': total, 'elegibles_fuente': elegibles_total,
        'filas_utilizadas': len(datos), 'fraccion': fraccion,
        'estados': {str(k): int(v) for k, v in estados.items()},
        'cortes': {'validacion': ANIO_CORTE_VALIDACION, 'prueba': ANIO_CORTE_PRUEBA},
        'fecha_actualizacion_estados': 'No confirmada; no inferirla del nombre del archivo',
    }
    return datos, auditoria, cobertura


def preparar_avance(datos):
    """Una única matriz sin indicadores ni LC para una comparación justa."""
    entrenar = datos['anio'] < ANIO_CORTE_VALIDACION
    validar = datos['anio'].between(ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA - 1)
    candidatas = ft.seleccionar_predictoras(datos, [], False)
    train = ft.preparar_matriz(datos.loc[entrenar, candidatas + ['fecha_prestamo']]).drop(columns='fecha_prestamo')
    valid = ft.preparar_matriz(datos.loc[validar, candidatas + ['fecha_prestamo']]).drop(columns='fecha_prestamo')
    columnas, _, decisiones = ft.filtrar_por_faltantes(
        train, list(train.columns), UMBRAL_FALTANTES, UMBRAL_REVISION)
    decisiones.loc[decisiones['decision'] == 'imputar + indicador', 'decision'] = 'imputar (baseline sin indicadores)'
    preprocesador, _ = ft.construir_preprocesador(train[columnas], [])
    X_train = preprocesador.fit_transform(train[columnas])
    X_valid = preprocesador.transform(valid[columnas])
    y_train = datos.loc[entrenar, 'default'].to_numpy()
    y_valid = datos.loc[validar, 'default'].to_numpy()
    assert np.isfinite(X_train).all() and np.isfinite(X_valid).all()
    assert set(np.unique(y_train)) == {0, 1} and set(np.unique(y_valid)) == {0, 1}
    assert not (entrenar & validar).any()
    return X_train, X_valid, y_train, y_valid, decisiones


def comparar_modelos(X_train, X_valid, y_train, y_valid):
    """Configuraciones iniciales fijas: no se busca el mejor modelo en prueba."""
    modelos = {
        'logistica_sin_indicadores': LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=SEMILLA),
        'arbol_decision': DecisionTreeClassifier(max_depth=6, min_samples_leaf=0.005, random_state=SEMILLA),
    }
    resultados, tiempos = {}, {}
    for nombre, modelo in modelos.items():
        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        tiempos[nombre] = time.perf_counter() - inicio
        score_valid = modelo.predict_proba(X_valid)[:, 1]
        umbral = ev.elegir_umbral(y_valid, score_valid)
        resultados[nombre] = {
            'entrenamiento': ev.evaluar(y_train, modelo.predict_proba(X_train)[:, 1], umbral),
            'validacion': ev.evaluar(y_valid, score_valid, umbral),
        }
    tasa = float(np.mean(y_train))
    resultados['clase_mayoritaria'] = {
        'entrenamiento': ev.evaluar(y_train, np.full(len(y_train), tasa), 1.1),
        'validacion': ev.evaluar(y_valid, np.full(len(y_valid), tasa), 1.1),
    }
    return resultados, tiempos


def guardar_avance(modo, auditoria, cobertura, decisiones, resultados, tiempos):
    salida = RAIZ / 'outputs' / 'avance' / modo
    salida.mkdir(parents=True, exist_ok=True)
    cobertura.to_csv(salida / 'cobertura_anio_plazo.csv')
    decisiones.to_csv(salida / 'faltantes_entrenamiento.csv')
    ev.guardar_metricas(resultados, salida / 'metrics.json', extra={
        'auditoria': auditoria, 'segundos_entrenamiento': tiempos,
        'configuracion': {'escenario': 'sin indicadores y sin LC', 'semilla': SEMILLA,
                         'arbol': {'max_depth': 6, 'min_samples_leaf': 0.005},
                         'prueba_evaluada': False},
        'versiones': {'python': platform.python_version(), 'pandas': pd.__version__,
                      'numpy': np.__version__, 'sklearn': sklearn.__version__},
    })
    figuras = RAIZ / 'reports' / 'figures' / 'avance' / modo
    figuras.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for plazo, grupo in cobertura.reset_index().groupby('plazo'):
        axes[0].plot(grupo['anio'], grupo['porcentaje_elegibles'], marker='o', label=f'{plazo:g} meses')
        axes[1].plot(grupo['anio'], grupo['porcentaje_incumplimiento_elegibles'], marker='o', label=f'{plazo:g} meses')
    axes[0].set_title('Porcentaje elegible entre préstamos con estado')
    axes[1].set_title('Incumplimiento entre los elegibles')
    for ax in axes:
        ax.set_xlabel('Año de otorgamiento'); ax.set_ylabel('%'); ax.legend()
    fig.tight_layout(); fig.savefig(figuras / 'cobertura.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    nombres = list(resultados)
    valores = [resultados[n]['validacion']['roc_auc'] for n in nombres]
    ax.barh(nombres, valores)
    ax.set_xlim(0, 1); ax.set_xlabel('ROC-AUC de validación')
    for i, v in enumerate(valores): ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    fig.tight_layout(); fig.savefig(figuras / 'comparacion.png', dpi=130); plt.close(fig)
    return salida


def ejecutar_avance(modo='muestra', ruta=None, tamano_muestra=TAMANO_MUESTRA):
    if modo not in ['muestra', 'completo']:
        raise ValueError('Modo esperado: muestra o completo.')
    print(f'Cargando modo {modo} por bloques...', flush=True)
    datos, auditoria, cobertura = cargar_avance(
        localizar_datos(ruta), None if modo == 'completo' else tamano_muestra)
    auditoria['particion'] = {
        'entrenamiento': int((datos['anio'] < ANIO_CORTE_VALIDACION).sum()),
        'validacion': int(datos['anio'].between(ANIO_CORTE_VALIDACION, ANIO_CORTE_PRUEBA - 1).sum()),
        'prueba_reservada': int((datos['anio'] >= ANIO_CORTE_PRUEBA).sum()),
    }
    print(auditoria['particion'], flush=True)
    print('Preparando las mismas variables para ambos modelos...', flush=True)
    X_train, X_valid, y_train, y_valid, decisiones = preparar_avance(datos)
    auditoria['columnas_modelo'] = X_train.shape[1]
    del datos
    gc.collect()
    print('Entrenando logística y árbol; prueba no se evalúa...', flush=True)
    resultados, tiempos = comparar_modelos(X_train, X_valid, y_train, y_valid)
    salida = guardar_avance(modo, auditoria, cobertura, decisiones, resultados, tiempos)
    print(ev.tabla_metricas(resultados).to_string(index=False), flush=True)
    print('Resultados guardados en', salida, flush=True)
    return resultados, auditoria


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modo', choices=['muestra', 'completo'], default='muestra')
    parser.add_argument('--datos', default=None)
    parser.add_argument('--muestra', type=int, default=TAMANO_MUESTRA)
    args = parser.parse_args()
    ejecutar_avance(args.modo, args.datos, args.muestra)
