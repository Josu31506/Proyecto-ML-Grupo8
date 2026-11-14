"""Experimento oficial: comparacion_imputacion. Conserva semillas, cortes y parámetros.
Las nuevas ejecuciones escriben en una carpeta independiente.
"""
from pathlib import Path

def ejecutar(raiz, salidas):
    from pathlib import Path
    raiz = Path(raiz).resolve()
    SALIDAS = Path(salidas).resolve()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import sys, time, gc
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

    ANIOS_VALIDACION = [2013, 2014, 2015, 2016]
    UMBRAL_FALTANTES = 10
    ruta_datos = localizar_datos()
    columnas_fuente = pd.read_csv(ruta_datos, nrows=0).columns.tolist()
    candidatas = ft.seleccionar_predictoras(
        pd.DataFrame(columns=columnas_fuente), [], incluir_precio_lc=False
    )
    columnas_lectura = candidatas + ["loan_status", "issue_d"]
    dtypes = {
        col: ("string" if TIPOS_AUDITADOS[col] in ["category", "string"] else TIPOS_AUDITADOS[col])
        for col in columnas_lectura
    }
    print(f"Candidatas antes del filtro: {len(candidatas)}")
    print("Leyendo por bloques; se conservan solo préstamos elegibles con issue_d hasta 2016...")
    partes=[]
    filas_fuente=0
    inicio=time.perf_counter()
    with pd.read_csv(ruta_datos, usecols=columnas_lectura, dtype=dtypes,
                     chunksize=100_000, skipinitialspace=True, low_memory=False) as lector:
        for numero,bloque in enumerate(lector, start=1):
            filas_fuente += len(bloque)
            fecha = pd.to_datetime(bloque["issue_d"], format="%b-%Y", errors="coerce")
            conservar = fecha.notna() & (fecha.dt.year <= 2016)
            bloque = bloque.loc[conservar].copy()
            bloque["issue_d"] = fecha.loc[conservar]
            bloque = bloque.loc[bloque["loan_status"].isin(MAPA_OBJETIVO)].copy()
            if bloque.empty:
                continue
            for col in ["earliest_cr_line", "sec_app_earliest_cr_line"]:
                if col in bloque:
                    original = bloque[col].notna()
                    fecha_col = pd.to_datetime(bloque[col], format="%b-%Y", errors="coerce")
                    if (original & fecha_col.isna()).any():
                        raise ValueError(f"Fecha no reconocida en {col}")
                    bloque[col] = fecha_col
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
                print(f"  Bloques recorridos: {numero}; filas fuente leídas: {filas_fuente:,}",flush=True)
    datos = pd.concat(partes,ignore_index=True)
    del partes
    datos = datos.loc[datos["anio"] <= 2016].copy()
    assert datos["anio"].max() <= 2016
    print(f"Datos de modelado: {len(datos):,} préstamos; años {datos.anio.min()}–{datos.anio.max()}; carga {time.perf_counter()-inicio:.1f} s")


    mask_2012 = datos["anio"] <= 2012
    faltantes_2012 = datos.loc[mask_2012,candidatas].isna().mean()*100
    variables_26 = faltantes_2012[faltantes_2012 < UMBRAL_FALTANTES].index.tolist()
    if len(variables_26) != 26:
        raise ValueError(f"Se esperaban 26 variables fijas, se obtuvieron {len(variables_26)}")
    print(f"Variables fijas: {len(variables_26)}; candidatas para selección temporal: {len(candidatas)}")


    from src.preparacion_temporal import preparar_X as preparar_X


    HIPERPARAMETROS = {
        "learning_rate": 0.08118441500491742,
        "max_iter": 182,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 46,
        "l2_regularization": 0.32877474139911184,
        "early_stopping": False,
        "random_state": SEMILLA,
    }

    def ajustar_y_evaluar(anio_validacion, variables, metodo, nombre_experimento):
        anio_final_train=anio_validacion-1
        mask_train=datos.anio<=anio_final_train
        mask_valid=datos.anio==anio_validacion
        X_train=preparar_X(datos.loc[mask_train,variables],datos.loc[mask_train,"fecha_prestamo"])
        X_valid=preparar_X(datos.loc[mask_valid,variables],datos.loc[mask_valid,"fecha_prestamo"])
        y_train=datos.loc[mask_train,"default"].to_numpy(dtype="int8")
        y_valid=datos.loc[mask_valid,"default"].to_numpy(dtype="int8")
        numericas=[col for col in X_train if pd.api.types.is_numeric_dtype(X_train[col])]
        categoricas=[col for col in X_train if col not in numericas]
        indicadores=[]

        if metodo=="mediana_indicadores":


            for col in numericas:
                if X_train[col].isna().any():
                    nombre=f"{col}__faltante"
                    X_train[nombre]=X_train[col].isna().astype("int8")
                    X_valid[nombre]=X_valid[col].isna().astype("int8")
                    indicadores.append(nombre)

        transformadores=[]
        if numericas:
            if metodo=="nan_nativo":
                pipeline_num=Pipeline([("escalar",StandardScaler())])
            else:
                pipeline_num=Pipeline([
                    ("imputar",SimpleImputer(strategy="median")),
                    ("escalar",StandardScaler()),
                ])
            transformadores.append(("numericas",pipeline_num,numericas))
        if categoricas:
            pipeline_cat=Pipeline([
                ("imputar",SimpleImputer(strategy="constant",fill_value="Sin información")),
                ("onehot",OneHotEncoder(handle_unknown="ignore",drop="first",
                                        sparse_output=False,dtype=np.float32)),
            ])
            transformadores.append(("categoricas",pipeline_cat,categoricas))
        if indicadores:
            transformadores.append(("indicadores","passthrough",indicadores))
        pre=ColumnTransformer(transformadores,remainder="drop",sparse_threshold=0.0)


        X_train_proc=pre.fit_transform(X_train)
        X_valid_proc=pre.transform(X_valid)
        nan_train_original=int(X_train[numericas].isna().to_numpy().sum()) if numericas else 0
        nan_valid_original=int(X_valid[numericas].isna().to_numpy().sum()) if numericas else 0
        if metodo=="nan_nativo":
            nan_train_transformado=int(np.isnan(X_train_proc).sum())
            nan_valid_transformado=int(np.isnan(X_valid_proc).sum())
            if nan_train_transformado != nan_train_original:
                raise AssertionError(f"El preprocesador alteró NaN de entrenamiento: {nan_train_original} -> {nan_train_transformado}")
            if nan_valid_transformado != nan_valid_original:
                raise AssertionError(f"El preprocesador alteró NaN de validación: {nan_valid_original} -> {nan_valid_transformado}")

        modelo=HistGradientBoostingClassifier(**HIPERPARAMETROS)
        inicio=time.perf_counter()
        modelo.fit(X_train_proc,y_train)
        segundos=time.perf_counter()-inicio
        puntuacion=modelo.predict_proba(X_valid_proc)[:,1]
        fila={
            "experimento":nombre_experimento,
            "metodo":metodo,
            "anio_final_entrenamiento":anio_final_train,
            "anio_validacion":anio_validacion,
            "filas_entrenamiento":len(y_train),
            "filas_validacion":len(y_valid),
            "variables_originales":len(variables),
            "indicadores_faltantes":len(indicadores),
            "variables_numericas":len(numericas),
            "variables_categoricas":len(categoricas),
            "faltantes_numericos_entrenamiento":nan_train_original,
            "faltantes_numericos_validacion":nan_valid_original,
            "faltante_maximo_pct_entrenamiento":float(datos.loc[mask_train,variables].isna().mean().max()*100),
            "tasa_incumplimiento_validacion":float(y_valid.mean()),
            "ROC_AUC":float(roc_auc_score(y_valid,puntuacion)),
            "PR_AUC":float(average_precision_score(y_valid,puntuacion)),
            "segundos_ajuste":float(segundos),
            "NaN_entrenamiento_conservados":(nan_train_original if metodo=="nan_nativo" else 0),
            "NaN_validacion_conservados":(nan_valid_original if metodo=="nan_nativo" else 0),
        }

        pct=datos.loc[mask_train,variables].isna().mean()*100
        columna_transformada={
            "earliest_cr_line":"meses_desde_earliest_cr_line",
            "sec_app_earliest_cr_line":"meses_desde_sec_app_earliest_cr_line",
        }
        cobertura=pd.DataFrame({
            "experimento":nombre_experimento,
            "anio_validacion":anio_validacion,
            "variable":variables,
            "faltante_pct_entrenamiento":[float(pct[v]) for v in variables],
            "indicador_creado":[bool(f"{columna_transformada.get(v,v)}__faltante" in indicadores) for v in variables],
        })
        del pre,modelo,X_train_proc,X_valid_proc,puntuacion,X_train,X_valid
        gc.collect()
        return fila,cobertura

    METODOS={
        "mediana":("A",False),
        "mediana_indicadores":("B",True),
        "nan_nativo":("C",False),
    }
    resultados_filas=[]
    cobertura_filas=[]



    mask_train_corte=datos.anio<=2012
    pct_corte=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_corte[pct_corte<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales:
        raise ValueError("No quedaron variables para este corte")

    conjuntos={
        "26_fijas":variables_26,
        "seleccion_temporal":variables_temporales,
    }
    for nombre_conjunto,variables in conjuntos.items():
        for metodo,(letra,_) in METODOS.items():
            experimento=f"{nombre_conjunto}_{metodo}"
            fila,cobertura=ajustar_y_evaluar(2013,variables,metodo,experimento)
            fila["seleccion"]="26 fijas" if nombre_conjunto=="26_fijas" else "solo entrenamiento"
            resultados_filas.append(fila)
            cobertura_filas.append(cobertura)
            print(f"{experimento}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}",flush=True)

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_imputacion_parcial.csv",index=False)
    print("Corte completado: entrenamiento hasta 2012, validación 2013.")



    mask_train_corte=datos.anio<=2013
    pct_corte=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_corte[pct_corte<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales:
        raise ValueError("No quedaron variables para este corte")

    conjuntos={
        "26_fijas":variables_26,
        "seleccion_temporal":variables_temporales,
    }
    for nombre_conjunto,variables in conjuntos.items():
        for metodo,(letra,_) in METODOS.items():
            experimento=f"{nombre_conjunto}_{metodo}"
            fila,cobertura=ajustar_y_evaluar(2014,variables,metodo,experimento)
            fila["seleccion"]="26 fijas" if nombre_conjunto=="26_fijas" else "solo entrenamiento"
            resultados_filas.append(fila)
            cobertura_filas.append(cobertura)
            print(f"{experimento}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}",flush=True)

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_imputacion_parcial.csv",index=False)
    print("Corte completado: entrenamiento hasta 2013, validación 2014.")



    mask_train_corte=datos.anio<=2014
    pct_corte=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_corte[pct_corte<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales:
        raise ValueError("No quedaron variables para este corte")

    conjuntos={
        "26_fijas":variables_26,
        "seleccion_temporal":variables_temporales,
    }
    for nombre_conjunto,variables in conjuntos.items():
        for metodo,(letra,_) in METODOS.items():
            experimento=f"{nombre_conjunto}_{metodo}"
            fila,cobertura=ajustar_y_evaluar(2015,variables,metodo,experimento)
            fila["seleccion"]="26 fijas" if nombre_conjunto=="26_fijas" else "solo entrenamiento"
            resultados_filas.append(fila)
            cobertura_filas.append(cobertura)
            print(f"{experimento}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}",flush=True)

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_imputacion_parcial.csv",index=False)
    print("Corte completado: entrenamiento hasta 2014, validación 2015.")



    mask_train_corte=datos.anio<=2015
    pct_corte=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_corte[pct_corte<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales:
        raise ValueError("No quedaron variables para este corte")

    conjuntos={
        "26_fijas":variables_26,
        "seleccion_temporal":variables_temporales,
    }
    for nombre_conjunto,variables in conjuntos.items():
        for metodo,(letra,_) in METODOS.items():
            experimento=f"{nombre_conjunto}_{metodo}"
            fila,cobertura=ajustar_y_evaluar(2016,variables,metodo,experimento)
            fila["seleccion"]="26 fijas" if nombre_conjunto=="26_fijas" else "solo entrenamiento"
            resultados_filas.append(fila)
            cobertura_filas.append(cobertura)
            print(f"{experimento}: {len(variables)} variables, {fila['indicadores_faltantes']} indicadores, ROC-AUC={fila['ROC_AUC']:.4f}, PR-AUC={fila['PR_AUC']:.4f}",flush=True)

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_imputacion_parcial.csv",index=False)
    print("Corte completado: entrenamiento hasta 2015, validación 2016.")


    resultados=pd.DataFrame(resultados_filas)
    cobertura=pd.concat(cobertura_filas,ignore_index=True)
    resumen=(resultados.groupby(["seleccion","experimento","metodo"],as_index=False)
        .agg(ROC_AUC_promedio=("ROC_AUC","mean"),ROC_AUC_desviacion=("ROC_AUC","std"),
             PR_AUC_promedio=("PR_AUC","mean"),PR_AUC_desviacion=("PR_AUC","std"),
             variables_min=("variables_originales","min"),variables_max=("variables_originales","max"),
             indicadores_promedio=("indicadores_faltantes","mean")))
    resumen[["ROC_AUC_desviacion","PR_AUC_desviacion"]]=resumen[["ROC_AUC_desviacion","PR_AUC_desviacion"]].fillna(0)

    filas_diferencias=[]
    for conjunto,base,indicadores,nativo in [
        ("26 fijas","26_fijas_mediana","26_fijas_mediana_indicadores","26_fijas_nan_nativo"),
        ("Selección temporal","seleccion_temporal_mediana","seleccion_temporal_mediana_indicadores","seleccion_temporal_nan_nativo"),
    ]:
        tbase=resultados[resultados.experimento==base]
        tind=resultados[resultados.experimento==indicadores]
        tnat=resultados[resultados.experimento==nativo]
        for metodo,tabla in [("Mediana + indicadores",tind),("NaN nativos",tnat)]:
            m=tbase[["anio_validacion","ROC_AUC","PR_AUC"]].merge(
                tabla[["anio_validacion","ROC_AUC","PR_AUC"]],on="anio_validacion",suffixes=("_mediana","_comparado"))
            filas_diferencias.append({
                "conjunto":conjunto,"metodo_comparado":metodo,
                "delta_ROC_AUC_promedio":float((m.ROC_AUC_comparado-m.ROC_AUC_mediana).mean()),
                "cortes_ROC_AUC_mejor":int((m.ROC_AUC_comparado>m.ROC_AUC_mediana).sum()),
                "delta_PR_AUC_promedio":float((m.PR_AUC_comparado-m.PR_AUC_mediana).mean()),
                "cortes_PR_AUC_mejor":int((m.PR_AUC_comparado>m.PR_AUC_mediana).sum()),
            })
    diferencias=pd.DataFrame(filas_diferencias)

    display(resumen.round(5))
    display(diferencias.round(5))
    display(resultados[["experimento","anio_validacion","variables_originales","indicadores_faltantes","faltantes_numericos_entrenamiento","NaN_entrenamiento_conservados","ROC_AUC","PR_AUC"]].sort_values(["experimento","anio_validacion"]).round(5))


    cobertura_resumen=(cobertura.groupby(["experimento","anio_validacion"],as_index=False)
        .agg(variables=("variable","count"),variables_con_faltantes=("faltante_pct_entrenamiento",lambda x:int((x>0).sum())),
             faltante_promedio_pct=("faltante_pct_entrenamiento","mean"),faltante_maximo_pct=("faltante_pct_entrenamiento","max"),
             indicadores=("indicador_creado","sum")))
    display(cobertura_resumen.round(2))


    orden=["26_fijas_mediana","26_fijas_mediana_indicadores","26_fijas_nan_nativo",
           "seleccion_temporal_mediana","seleccion_temporal_mediana_indicadores","seleccion_temporal_nan_nativo"]
    resplot=resumen.set_index("experimento").reindex(orden)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,metrica,desv,titulo in [
        (axes[0],"ROC_AUC_promedio","ROC_AUC_desviacion","ROC-AUC promedio"),
        (axes[1],"PR_AUC_promedio","PR_AUC_desviacion","PR-AUC (Average Precision) promedio")]:
        ax.bar(range(len(orden)),resplot[metrica],yerr=resplot[desv],capsize=4,
               color=["#4C78A8","#72A0C1","#93B5D1","#54A24B","#8BC28A","#B2D9B0"])
        ax.set_xticks(range(len(orden)),[x.replace("_","\n") for x in orden],rotation=20,ha="right")
        ax.set_title(titulo+" (barras: desviación entre años)")
        ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(SALIDAS / f"figura_{fig.number}.png", dpi=150, bbox_inches="tight"); display(fig); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for experimento in orden:
        sub=resultados[resultados.experimento==experimento].sort_values("anio_validacion")
        axes[0].plot(sub.anio_validacion,sub.ROC_AUC,marker="o",label=experimento)
        axes[1].plot(sub.anio_validacion,sub.PR_AUC,marker="o",label=experimento)
    axes[0].set_title("ROC-AUC por año"); axes[1].set_title("PR-AUC por año")
    for ax in axes:
        ax.set_xticks(ANIOS_VALIDACION); ax.set_xlabel("Año de validación"); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8,bbox_to_anchor=(1.02,1),loc="upper left")
    fig.tight_layout(); fig.savefig(SALIDAS / f"figura_{fig.number}.png", dpi=150, bbox_inches="tight"); display(fig); plt.close(fig)

    salida=SALIDAS/"comparacion_imputacion"
    salida.mkdir(parents=True,exist_ok=True)
    resultados.to_csv(salida/"metricas_por_experimento_y_anio.csv",index=False)
    resumen.to_csv(salida/"resumen_promedios_y_variabilidad.csv",index=False)
    diferencias.to_csv(salida/"diferencias_frente_mediana.csv",index=False)
    cobertura.to_csv(salida/"faltantes_variables_por_corte.csv",index=False)
    cobertura_resumen.to_csv(salida/"resumen_faltantes_por_corte.csv",index=False)
    pd.DataFrame({"variable_26":pd.Series(variables_26),"candidatas":pd.Series(candidatas)}).to_csv(salida/"listas_variables.csv",index=False)
    print("Resultados guardados en:",salida)
    for _,r in diferencias.iterrows():
        print(f"{r['conjunto']} — {r['metodo_comparado']}: delta ROC-AUC {r['delta_ROC_AUC_promedio']:+.5f} ({int(r['cortes_ROC_AUC_mejor'])}/4 cortes mejores); delta PR-AUC {r['delta_PR_AUC_promedio']:+.5f} ({int(r['cortes_PR_AUC_mejor'])}/4 cortes mejores).")

    return {k: v for k, v in locals().items() if k in ("resumen", "resultados", "resumen_modelos", "resultados_por_corte")}
