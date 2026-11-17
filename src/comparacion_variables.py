"""Experimento oficial: comparacion_variables. Conserva semillas, cortes y parámetros.
Las nuevas ejecuciones escriben en una carpeta independiente.
"""
from pathlib import Path

def ejecutar(raiz, salidas):
    from pathlib import Path
    raiz = Path(raiz).resolve()
    SALIDAS = Path(salidas).resolve()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import sys, time, gc, json, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    RAIZ = raiz
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))

    from src.config import localizar_datos, MAPA_OBJETIVO, SEMILLA
    from src.modelado import TIPOS_AUDITADOS
    from src import features as ft

    SEMILLA = 42
    ANIOS_VALIDACION = [2013, 2014, 2015, 2016]
    UMBRAL_FALTANTES = 10

    ruta_datos = localizar_datos()
    columnas_fuente = pd.read_csv(ruta_datos, nrows=0).columns.tolist()
    candidatas = ft.seleccionar_predictoras(
        pd.DataFrame(columns=columnas_fuente), [], incluir_precio_lc=False
    )
    columnas_lectura = candidatas + ["loan_status", "issue_d"]
    dtypes = {
        c: ("string" if TIPOS_AUDITADOS[c] in ["category", "string"] else TIPOS_AUDITADOS[c])
        for c in columnas_lectura
    }
    print(f"Candidatas previas al filtro: {len(candidatas)}")
    print("Leyendo por bloques y reteniendo solo préstamos elegibles hasta 2016...")

    partes = []
    filas_leidas = 0
    inicio_carga = time.perf_counter()
    with pd.read_csv(ruta_datos, usecols=columnas_lectura, dtype=dtypes,
                     chunksize=100_000, skipinitialspace=True, low_memory=False) as lector:
        for numero, bloque in enumerate(lector, start=1):
            filas_leidas += len(bloque)
            fecha = pd.to_datetime(bloque["issue_d"], format="%b-%Y", errors="coerce")

            conservar_fecha = fecha.notna() & (fecha.dt.year <= 2016)
            bloque = bloque.loc[conservar_fecha].copy()
            bloque["issue_d"] = fecha.loc[conservar_fecha]
            elegibles = bloque["loan_status"].isin(MAPA_OBJETIVO)
            bloque = bloque.loc[elegibles].copy()
            if bloque.empty:
                continue
            for col in ["earliest_cr_line", "sec_app_earliest_cr_line"]:
                if col in bloque:
                    original = bloque[col].notna()
                    convertida = pd.to_datetime(bloque[col], format="%b-%Y", errors="coerce")
                    if (original & convertida.isna()).any():
                        raise ValueError(f"Fecha no reconocida en {col}")
                    bloque[col] = convertida
            if "term" in bloque:
                bloque["term"] = pd.to_numeric(
                    bloque["term"].astype("string").str.strip().str.replace(" months", "", regex=False),
                    errors="raise"
                ).astype("float64")
            bloque["anio"] = bloque["issue_d"].dt.year.astype("int16")
            bloque["default"] = bloque["loan_status"].map(MAPA_OBJETIVO).astype("int8")
            bloque["fecha_prestamo"] = bloque["issue_d"]
            partes.append(bloque)
            if numero % 10 == 0:
                print(f"  Bloques leídos: {numero}; filas fuente recorridas: {filas_leidas:,}", flush=True)

    datos = pd.concat(partes, ignore_index=True)
    del partes
    print(f"Datos de análisis: {len(datos):,} préstamos elegibles hasta 2016; carga {time.perf_counter()-inicio_carga:.1f} s")
    print("Años incluidos:", int(datos["anio"].min()), "a", int(datos["anio"].max()))
    assert datos["anio"].max() <= 2016


    mascara_2012 = datos["anio"] <= 2012
    faltantes_hasta_2012 = datos.loc[mascara_2012, candidatas].isna().mean() * 100
    variables_26 = faltantes_hasta_2012[faltantes_hasta_2012 < UMBRAL_FALTANTES].index.tolist()


    faltantes_hasta_2016 = datos[candidatas].isna().mean() * 100
    variables_59 = faltantes_hasta_2016[faltantes_hasta_2016 < UMBRAL_FALTANTES].index.tolist()

    print(f"A/B: {len(variables_26)} variables seleccionadas hasta 2012")
    print(f"C/D: {len(variables_59)} variables seleccionadas hasta 2016 (exploratorio)")
    print(f"Candidatas para E/F: {len(candidatas)}")
    if len(variables_26) != 26 or len(variables_59) != 59:
        raise ValueError(f"Se esperaba 26 y 59 variables; se obtuvieron {len(variables_26)} y {len(variables_59)}")

    tabla_resumen_seleccion = pd.DataFrame({
        "conjunto": ["Candidatas", "26 fijas (hasta 2012)", "59 fijas (hasta 2016, exploratorio)"],
        "cantidad": [len(candidatas), len(variables_26), len(variables_59)],
    })
    display(tabla_resumen_seleccion)


    from src.preparacion_temporal import preparar_X as derivar_antiguedad



    HIPERPARAMETROS = {
        "learning_rate": 0.08118441500491742,
        "max_iter": 182,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 46,
        "l2_regularization": 0.32877474139911184,
        "early_stopping": False,
        "random_state": SEMILLA,
    }

    def evaluar_corte(anio_validacion, variables, usar_indicadores, nombre_experimento):
        anio_final_train = anio_validacion - 1
        mascara_train = datos["anio"] <= anio_final_train
        mascara_valid = datos["anio"] == anio_validacion
        fecha_train = datos.loc[mascara_train, "fecha_prestamo"]
        fecha_valid = datos.loc[mascara_valid, "fecha_prestamo"]
        X_train = derivar_antiguedad(datos.loc[mascara_train, variables], fecha_train)
        X_valid = derivar_antiguedad(datos.loc[mascara_valid, variables], fecha_valid)
        y_train = datos.loc[mascara_train, "default"].to_numpy(dtype="int8")
        y_valid = datos.loc[mascara_valid, "default"].to_numpy(dtype="int8")

        numericas = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        categoricas = [c for c in X_train.columns if c not in numericas]
        nombres_indicadores = []
        if usar_indicadores:

            for col in numericas:
                if X_train[col].isna().any():
                    nombre = f"{col}__faltante"
                    if nombre in X_train.columns:
                        raise ValueError(f"Colisión de nombre para indicador: {nombre}")
                    X_train[nombre] = X_train[col].isna().astype("int8")
                    X_valid[nombre] = X_valid[col].isna().astype("int8")
                    nombres_indicadores.append(nombre)

        transformadores = []
        if numericas:
            transformadores.append((
                "numericas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler()),
                ]),
                numericas,
            ))
        if categoricas:
            transformadores.append((
                "categoricas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="constant", fill_value="Sin información")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", drop="first", sparse_output=False,
                        dtype=np.float32,
                    )),
                ]),
                categoricas,
            ))
        if nombres_indicadores:
            transformadores.append(("indicadores", "passthrough", nombres_indicadores))
        preprocesador = ColumnTransformer(
            transformadores, remainder="drop", sparse_threshold=0.0
        )
        modelo = HistGradientBoostingClassifier(**HIPERPARAMETROS)
        pipeline = Pipeline([("preprocesar", preprocesador), ("modelo", modelo)])
        inicio = time.perf_counter()
        pipeline.fit(X_train, y_train)
        segundos = time.perf_counter() - inicio
        puntuacion = pipeline.predict_proba(X_valid)[:, 1]

        roc = roc_auc_score(y_valid, puntuacion)
        ap = average_precision_score(y_valid, puntuacion)
        faltantes_train = datos.loc[mascara_train, variables].isna().mean() * 100
        detalle_faltantes = pd.DataFrame({
            "experimento": nombre_experimento,
            "anio_validacion": anio_validacion,
            "variable": variables,
            "faltante_pct_entrenamiento": [float(faltantes_train.get(v, np.nan)) for v in variables],
            "indicador_creado": [bool(f"{v}__faltante" in nombres_indicadores) for v in variables],
        })
        fila = {
            "experimento": nombre_experimento,
            "anio_final_entrenamiento": anio_final_train,
            "anio_validacion": anio_validacion,
            "filas_entrenamiento": len(y_train),
            "filas_validacion": len(y_valid),
            "variables_originales": len(variables),
            "indicadores_faltantes": len(nombres_indicadores),
            "variables_numericas": len(numericas),
            "variables_categoricas": len(categoricas),
            "faltante_promedio_pct_entrenamiento": float(faltantes_train.mean()),
            "faltante_maximo_pct_entrenamiento": float(faltantes_train.max()),
            "tasa_incumplimiento_validacion": float(y_valid.mean()),
            "ROC_AUC": float(roc),
            "PR_AUC": float(ap),
            "segundos_ajuste": float(segundos),
        }
        del pipeline, puntuacion, X_train, X_valid
        gc.collect()
        return fila, detalle_faltantes

    experimentos = {
        "A_26_sin_indicadores": ("26 fijas", variables_26, False, "rigurosa"),
        "B_26_con_indicadores": ("26 fijas", variables_26, True, "rigurosa"),
        "C_59_sin_indicadores": ("59 fijas", variables_59, False, "exploratoria"),
        "D_59_con_indicadores": ("59 fijas", variables_59, True, "exploratoria"),
    }
    filas_resultados = []
    filas_faltantes = []



    mascara_train = datos["anio"] <= 2012
    faltantes_train_corte = datos.loc[mascara_train, candidatas].isna().mean() * 100
    variables_corte = faltantes_train_corte[faltantes_train_corte < UMBRAL_FALTANTES].index.tolist()
    if not variables_corte:
        raise ValueError("El corte no dejó variables predictoras.")

    experimentos_corte = {
        "A_26_sin_indicadores": variables_26,
        "B_26_con_indicadores": variables_26,
        "C_59_sin_indicadores": variables_59,
        "D_59_con_indicadores": variables_59,
        "E_temporal_sin_indicadores": variables_corte,
        "F_temporal_con_indicadores": variables_corte,
    }
    config_indicadores = {
        "A_26_sin_indicadores": False,
        "B_26_con_indicadores": True,
        "C_59_sin_indicadores": False,
        "D_59_con_indicadores": True,
        "E_temporal_sin_indicadores": False,
        "F_temporal_con_indicadores": True,
    }

    for nombre, variables in experimentos_corte.items():
        fila, detalle = evaluar_corte(2013, variables, config_indicadores[nombre], nombre)
        fila["tipo_seleccion"] = "exploratoria" if nombre.startswith(("C_", "D_")) else "solo_entrenamiento"
        filas_resultados.append(fila)
        filas_faltantes.append(detalle)
        print(f"{nombre}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}", flush=True)

    tabla_parcial = pd.DataFrame(filas_resultados)
    tabla_parcial.to_csv(SALIDAS / "comparacion_variables_temporal_parcial.csv", index=False)
    print("Corte completado: entrenamiento hasta 2012, validación 2013.")



    mascara_train = datos["anio"] <= 2013
    faltantes_train_corte = datos.loc[mascara_train, candidatas].isna().mean() * 100
    variables_corte = faltantes_train_corte[faltantes_train_corte < UMBRAL_FALTANTES].index.tolist()
    if not variables_corte:
        raise ValueError("El corte no dejó variables predictoras.")

    experimentos_corte = {
        "A_26_sin_indicadores": variables_26,
        "B_26_con_indicadores": variables_26,
        "C_59_sin_indicadores": variables_59,
        "D_59_con_indicadores": variables_59,
        "E_temporal_sin_indicadores": variables_corte,
        "F_temporal_con_indicadores": variables_corte,
    }
    config_indicadores = {
        "A_26_sin_indicadores": False,
        "B_26_con_indicadores": True,
        "C_59_sin_indicadores": False,
        "D_59_con_indicadores": True,
        "E_temporal_sin_indicadores": False,
        "F_temporal_con_indicadores": True,
    }

    for nombre, variables in experimentos_corte.items():
        fila, detalle = evaluar_corte(2014, variables, config_indicadores[nombre], nombre)
        fila["tipo_seleccion"] = "exploratoria" if nombre.startswith(("C_", "D_")) else "solo_entrenamiento"
        filas_resultados.append(fila)
        filas_faltantes.append(detalle)
        print(f"{nombre}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}", flush=True)

    tabla_parcial = pd.DataFrame(filas_resultados)
    tabla_parcial.to_csv(SALIDAS / "comparacion_variables_temporal_parcial.csv", index=False)
    print("Corte completado: entrenamiento hasta 2013, validación 2014.")



    mascara_train = datos["anio"] <= 2014
    faltantes_train_corte = datos.loc[mascara_train, candidatas].isna().mean() * 100
    variables_corte = faltantes_train_corte[faltantes_train_corte < UMBRAL_FALTANTES].index.tolist()
    if not variables_corte:
        raise ValueError("El corte no dejó variables predictoras.")

    experimentos_corte = {
        "A_26_sin_indicadores": variables_26,
        "B_26_con_indicadores": variables_26,
        "C_59_sin_indicadores": variables_59,
        "D_59_con_indicadores": variables_59,
        "E_temporal_sin_indicadores": variables_corte,
        "F_temporal_con_indicadores": variables_corte,
    }
    config_indicadores = {
        "A_26_sin_indicadores": False,
        "B_26_con_indicadores": True,
        "C_59_sin_indicadores": False,
        "D_59_con_indicadores": True,
        "E_temporal_sin_indicadores": False,
        "F_temporal_con_indicadores": True,
    }

    for nombre, variables in experimentos_corte.items():
        fila, detalle = evaluar_corte(2015, variables, config_indicadores[nombre], nombre)
        fila["tipo_seleccion"] = "exploratoria" if nombre.startswith(("C_", "D_")) else "solo_entrenamiento"
        filas_resultados.append(fila)
        filas_faltantes.append(detalle)
        print(f"{nombre}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}", flush=True)

    tabla_parcial = pd.DataFrame(filas_resultados)
    tabla_parcial.to_csv(SALIDAS / "comparacion_variables_temporal_parcial.csv", index=False)
    print("Corte completado: entrenamiento hasta 2014, validación 2015.")



    mascara_train = datos["anio"] <= 2015
    faltantes_train_corte = datos.loc[mascara_train, candidatas].isna().mean() * 100
    variables_corte = faltantes_train_corte[faltantes_train_corte < UMBRAL_FALTANTES].index.tolist()
    if not variables_corte:
        raise ValueError("El corte no dejó variables predictoras.")

    experimentos_corte = {
        "A_26_sin_indicadores": variables_26,
        "B_26_con_indicadores": variables_26,
        "C_59_sin_indicadores": variables_59,
        "D_59_con_indicadores": variables_59,
        "E_temporal_sin_indicadores": variables_corte,
        "F_temporal_con_indicadores": variables_corte,
    }
    config_indicadores = {
        "A_26_sin_indicadores": False,
        "B_26_con_indicadores": True,
        "C_59_sin_indicadores": False,
        "D_59_con_indicadores": True,
        "E_temporal_sin_indicadores": False,
        "F_temporal_con_indicadores": True,
    }

    for nombre, variables in experimentos_corte.items():
        fila, detalle = evaluar_corte(2016, variables, config_indicadores[nombre], nombre)
        fila["tipo_seleccion"] = "exploratoria" if nombre.startswith(("C_", "D_")) else "solo_entrenamiento"
        filas_resultados.append(fila)
        filas_faltantes.append(detalle)
        print(f"{nombre}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}", flush=True)

    tabla_parcial = pd.DataFrame(filas_resultados)
    tabla_parcial.to_csv(SALIDAS / "comparacion_variables_temporal_parcial.csv", index=False)
    print("Corte completado: entrenamiento hasta 2015, validación 2016.")


    resultados = pd.DataFrame(filas_resultados)
    faltantes_detalle = pd.concat(filas_faltantes, ignore_index=True)

    resumen = (resultados.groupby(["experimento", "tipo_seleccion"], as_index=False)
               .agg(ROC_AUC_promedio=("ROC_AUC", "mean"),
                    ROC_AUC_desviacion=("ROC_AUC", "std"),
                    PR_AUC_promedio=("PR_AUC", "mean"),
                    PR_AUC_desviacion=("PR_AUC", "std"),
                    variables_min=("variables_originales", "min"),
                    variables_max=("variables_originales", "max"),
                    indicadores_promedio=("indicadores_faltantes", "mean")))
    resumen["ROC_AUC_desviacion"] = resumen["ROC_AUC_desviacion"].fillna(0)
    resumen["PR_AUC_desviacion"] = resumen["PR_AUC_desviacion"].fillna(0)

    pares = [
        ("26 variables", "A_26_sin_indicadores", "B_26_con_indicadores"),
        ("59 variables (exploratorio)", "C_59_sin_indicadores", "D_59_con_indicadores"),
        ("Selección por corte", "E_temporal_sin_indicadores", "F_temporal_con_indicadores"),
    ]
    filas_pares=[]
    for etiqueta, sin_ind, con_ind in pares:
        a = resultados.loc[resultados.experimento == sin_ind]
        b = resultados.loc[resultados.experimento == con_ind]
        merge = a[["anio_validacion", "ROC_AUC", "PR_AUC"]].merge(
            b[["anio_validacion", "ROC_AUC", "PR_AUC"]], on="anio_validacion", suffixes=("_sin", "_con"))
        filas_pares.append({
            "conjunto": etiqueta,
            "ROC_AUC_sin_indicadores": merge.ROC_AUC_sin.mean(),
            "ROC_AUC_con_indicadores": merge.ROC_AUC_con.mean(),
            "diferencia_ROC_AUC": (merge.ROC_AUC_con-merge.ROC_AUC_sin).mean(),
            "cortes_ROC_AUC_mejora": int((merge.ROC_AUC_con>merge.ROC_AUC_sin).sum()),
            "PR_AUC_sin_indicadores": merge.PR_AUC_sin.mean(),
            "PR_AUC_con_indicadores": merge.PR_AUC_con.mean(),
            "diferencia_PR_AUC": (merge.PR_AUC_con-merge.PR_AUC_sin).mean(),
            "cortes_PR_AUC_mejora": int((merge.PR_AUC_con>merge.PR_AUC_sin).sum()),
        })
    comparacion_indicadores = pd.DataFrame(filas_pares)

    display(resumen.round(5))
    display(comparacion_indicadores.round(5))

    display(resultados[["experimento", "anio_validacion", "variables_originales", "indicadores_faltantes", "ROC_AUC", "PR_AUC"]].sort_values(["experimento","anio_validacion"]).round(5))


    display(faltantes_detalle.sort_values(
        ["experimento", "anio_validacion", "faltante_pct_entrenamiento"],
        ascending=[True, True, False]
    ).groupby(["experimento", "anio_validacion"], as_index=False).head(5).round(2))


    orden = ["A_26_sin_indicadores", "B_26_con_indicadores", "C_59_sin_indicadores",
             "D_59_con_indicadores", "E_temporal_sin_indicadores", "F_temporal_con_indicadores"]
    resumen_plot = resumen.set_index("experimento").reindex(orden)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metrica, desviacion, titulo in [
        (axes[0], "ROC_AUC_promedio", "ROC_AUC_desviacion", "ROC-AUC promedio"),
        (axes[1], "PR_AUC_promedio", "PR_AUC_desviacion", "PR-AUC (Average Precision) promedio"),
    ]:
        ax.bar(range(len(orden)), resumen_plot[metrica], yerr=resumen_plot[desviacion], capsize=4,
               color=["#4C78A8", "#72A0C1", "#F2A541", "#F6C477", "#54A24B", "#8BC28A"])
        ax.set_xticks(range(len(orden)), [x.replace("_", "\n") for x in orden], rotation=25, ha="right")
        ax.set_title(titulo + " (barras: desviación entre años)")
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(SALIDAS / f"figura_{fig.number}.png", dpi=150, bbox_inches="tight"); display(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nombre in orden:
        subset = resultados[resultados.experimento == nombre].sort_values("anio_validacion")
        axes[0].plot(subset.anio_validacion, subset.ROC_AUC, marker="o", label=nombre)
        axes[1].plot(subset.anio_validacion, subset.PR_AUC, marker="o", label=nombre)
    axes[0].set_title("ROC-AUC por año")
    axes[1].set_title("PR-AUC por año")
    for ax in axes:
        ax.set_xticks(ANIOS_VALIDACION)
        ax.set_xlabel("Año de validación")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(SALIDAS / f"figura_{fig.number}.png", dpi=150, bbox_inches="tight"); display(fig)
    plt.close(fig)

    salida = SALIDAS / "comparacion_variables_temporal"
    salida.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(salida / "metricas_por_experimento_y_anio.csv", index=False)
    resumen.to_csv(salida / "resumen_promedios_y_variabilidad.csv", index=False)
    comparacion_indicadores.to_csv(salida / "comparacion_indicadores.csv", index=False)
    faltantes_detalle.to_csv(salida / "faltantes_por_variable_y_corte.csv", index=False)
    pd.DataFrame({"variable_26": pd.Series(variables_26), "variable_59": pd.Series(variables_59)}).to_csv(salida / "listas_26_y_59.csv", index=False)
    print("Resultados guardados en:", salida)


    idx = resumen.set_index("experimento")
    delta_59 = idx.loc["C_59_sin_indicadores", "ROC_AUC_promedio"] - idx.loc["A_26_sin_indicadores", "ROC_AUC_promedio"]
    delta_temporal = idx.loc["E_temporal_sin_indicadores", "ROC_AUC_promedio"] - idx.loc["A_26_sin_indicadores", "ROC_AUC_promedio"]
    print(f"59 exploratorias frente a 26 sin indicadores: Δ ROC-AUC={delta_59:+.5f}.")
    print(f"Selección por corte frente a 26 sin indicadores: Δ ROC-AUC={delta_temporal:+.5f}.")
    for _, fila in comparacion_indicadores.iterrows():
        print(f"Indicadores en {fila['conjunto']}: Δ ROC-AUC={fila['diferencia_ROC_AUC']:+.5f} (mejora en {int(fila['cortes_ROC_AUC_mejora'])}/4 cortes); Δ PR-AUC={fila['diferencia_PR_AUC']:+.5f} (mejora en {int(fila['cortes_PR_AUC_mejora'])}/4 cortes).")

    return {k: v for k, v in locals().items() if k in ("resumen", "resultados", "resumen_modelos", "resultados_por_corte")}
