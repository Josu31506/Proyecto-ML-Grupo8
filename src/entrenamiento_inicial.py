"""Experimento oficial: entrenamiento_inicial. Conserva semillas, cortes y parámetros.
Las nuevas ejecuciones escriben en una carpeta independiente.
"""
from pathlib import Path

def ejecutar(raiz, salidas):
    from pathlib import Path
    raiz = Path(raiz).resolve()
    SALIDAS = Path(salidas).resolve()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import sys
    import time
    import gc
    import pandas as pd
    import numpy as np
    from IPython.display import display

    RAIZ = raiz
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))

    from src.config import localizar_datos, ANIO_CORTE_PRUEBA
    from src.modelado import (
        NOMBRES_VARIABLES,
        cargar_datos_modelado,
        crear_cortes_temporales,
        crear_tablas_imputadas_corte,
        modelos_iniciales,
        evaluar_modelos_temporales,
        resumir_metricas,
    )

    archivo = localizar_datos()
    print("Archivo fuente:", archivo)
    inicio = time.perf_counter()
    datos, predictoras = cargar_datos_modelado(archivo)
    tiempo_carga = time.perf_counter() - inicio
    memoria = datos.memory_usage(index=True, deep=True).sum() / 1024**3
    print(f"Registros elegibles cargados: {len(datos):,}")
    print(f"Predictoras conservadas: {len(predictoras)}")
    print(f"Columnas en la tabla de modelado (predictoras + fecha/año/objetivo): {datos.shape[1]}")
    print(f"Tiempo de carga y preparación: {tiempo_carga:.1f} segundos")
    print(f"Memoria profunda de la tabla: {memoria:.2f} GiB")
    print("Criterio aplicado: menos del 10% de faltantes en el primer entrenamiento (hasta 2012).")

    distribucion = datos["default"].value_counts().sort_index().rename(
        index={0: "Fully Paid", 1: "Charged Off"}
    ).to_frame("préstamos")
    distribucion["porcentaje"] = distribucion["préstamos"] / len(datos) * 100
    display(distribucion)


    tabla_cortes = crear_cortes_temporales(
        datos,
        primer_anio_validacion=2013,
        ultimo_anio_validacion=2016,
        anio_prueba=ANIO_CORTE_PRUEBA,
    )
    display(tabla_cortes)


    corte_ejemplo = crear_tablas_imputadas_corte(
        datos, predictoras, anio_final_entrenamiento=2012,
        anio_validacion=2013, usar_indicadores=True,
    )
    corte_ejemplo_sin_indicadores = crear_tablas_imputadas_corte(
        datos, predictoras, anio_final_entrenamiento=2012,
        anio_validacion=2013, usar_indicadores=False,
    )

    print("Con indicadores — X entrenamiento:", corte_ejemplo["X_entrenamiento"].shape)
    print("Sin indicadores — X entrenamiento:", corte_ejemplo_sin_indicadores["X_entrenamiento"].shape)
    print("Filas idénticas:", corte_ejemplo["X_entrenamiento"].index.equals(corte_ejemplo_sin_indicadores["X_entrenamiento"].index))
    print("Indicadores numéricos creados:", len(corte_ejemplo["columnas_indicadoras"]))
    print("Faltantes restantes en X entrenamiento:", int(corte_ejemplo["X_entrenamiento"].isna().sum().sum()))
    print("Faltantes restantes en X validación:", int(corte_ejemplo["X_validacion"].isna().sum().sum()))

    medianas_ejemplo = corte_ejemplo["medianas"].rename(index=lambda c: NOMBRES_VARIABLES.get(c, c)).to_frame(
        "mediana_calculada_hasta_2012"
    )
    print("Medianas aprendidas usando solo el entrenamiento:")
    display(medianas_ejemplo)
    print("Primeras filas imputadas del entrenamiento:")
    display(corte_ejemplo["X_entrenamiento"].head().rename(columns=lambda c: NOMBRES_VARIABLES.get(c, c)))
    print("Primeras filas imputadas de validación:")
    display(corte_ejemplo["X_validacion"].head().rename(columns=lambda c: NOMBRES_VARIABLES.get(c, c)))


    del corte_ejemplo, corte_ejemplo_sin_indicadores, medianas_ejemplo
    gc.collect()


    modelos = modelos_iniciales()
    print("Modelos que se evaluarán:")
    for nombre, modelo in modelos.items():
        print(f"- {nombre}: {modelo}")


    inicio_modelos = time.perf_counter()
    resultados_variantes = []
    tablas_medianas = []
    for usar_indicadores in [False, True]:
        nombre_variante = "mediana + indicadores" if usar_indicadores else "mediana sola"
        print(f"Evaluando {nombre_variante}.", flush=True)
        resultado_variante, medianas_variante = evaluar_modelos_temporales(
            datos, predictoras, modelos=modelos,
            primer_anio_validacion=2013, ultimo_anio_validacion=2016,
            usar_indicadores=usar_indicadores,
            umbral_clasificacion=0.5, verbose=True,
        )
        resultados_variantes.append(resultado_variante)
        if not usar_indicadores:
            tablas_medianas = medianas_variante
        else:
            pd.testing.assert_frame_equal(tablas_medianas, medianas_variante)
        del resultado_variante, medianas_variante
        gc.collect()
    resultados_por_corte = pd.concat(resultados_variantes, ignore_index=True)
    tiempo_modelos = time.perf_counter() - inicio_modelos

    columnas_metricas = [
        "modelo", "indicadores_faltantes", "corte", "n", "tasa_default_real",
        "roc_auc", "pr_auc", "ks", "recall", "precision", "f1", "accuracy",
        "segundos_entrenamiento",
    ]
    print(f"Tiempo total de entrenamiento y evaluación: {tiempo_modelos / 60:.1f} minutos")
    print("Resultados por modelo, variante de faltantes y año:")
    display(resultados_por_corte[columnas_metricas].round(4))

    resumen_modelos = resumir_metricas(resultados_por_corte)
    print("Promedio y variación entre los cuatro años de validación:")
    display(resumen_modelos.round(4))

    print("Medianas utilizadas por cada corte de entrenamiento:")
    tablas_medianas.index = [NOMBRES_VARIABLES.get(c, c) for c in tablas_medianas.index]
    display(tablas_medianas.round(2))

    salida = SALIDAS / "entrenamiento_26_variables"
    salida.mkdir(parents=True, exist_ok=True)
    resultados_por_corte.to_csv(salida / "metricas_validacion_temporal.csv", index=False, encoding="utf-8-sig")
    resumen_modelos.to_csv(salida / "resumen_modelos_validacion.csv", encoding="utf-8-sig")
    tablas_medianas.to_csv(salida / "medianas_por_corte.csv", encoding="utf-8-sig")
    print("Resultados guardados en:", salida)


    from pathlib import Path
    import pandas as pd
    from IPython.display import display
    RAIZ = raiz
    carpeta_resultados = SALIDAS / "entrenamiento_26_variables"
    resumen_guardado = pd.read_csv(carpeta_resultados / "resumen_modelos_validacion.csv", index_col=0)
    metricas_guardadas = pd.read_csv(carpeta_resultados / "metricas_validacion_temporal.csv", encoding="utf-8-sig")
    display(resumen_guardado.round(4))
    display(metricas_guardadas[["modelo", "indicadores_faltantes", "anio_validacion", "roc_auc", "pr_auc", "segundos_entrenamiento"]].round(4))
    return {k: v for k, v in locals().items() if k in ("resumen", "resultados", "resumen_modelos", "resultados_por_corte")}
