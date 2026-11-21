"""Experimento oficial: comparacion_lineales. Conserva semillas, cortes y parámetros.
Las nuevas ejecuciones escriben en una carpeta independiente.
"""
from pathlib import Path

def ejecutar(raiz, salidas, ruta_datos=None, referencias=None):
    from pathlib import Path
    raiz = Path(raiz).resolve()
    SALIDAS = Path(salidas).resolve()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import sys,time,gc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from scipy import sparse
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import roc_auc_score,average_precision_score
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    RAIZ = raiz
    if str(RAIZ) not in sys.path: sys.path.insert(0,str(RAIZ))
    from src.config import localizar_datos,MAPA_OBJETIVO,SEMILLA
    from src.modelado import TIPOS_AUDITADOS
    from src import features as ft

    ANIOS_VALIDACION=[2013,2014,2015,2016]
    UMBRAL_FALTANTES=10
    RUTA_DATOS=localizar_datos(ruta_datos)
    columnas_fuente=pd.read_csv(RUTA_DATOS,nrows=0).columns.tolist()
    candidatas=ft.seleccionar_predictoras(pd.DataFrame(columns=columnas_fuente),[],incluir_precio_lc=False)
    cols_lectura=candidatas+["loan_status","issue_d"]
    dtypes={col:("string" if TIPOS_AUDITADOS[col] in ["category","string"] else TIPOS_AUDITADOS[col]) for col in cols_lectura}
    print(f"Predictoras candidatas previas al filtro: {len(candidatas)}")
    print("Leyendo por bloques; se retienen préstamos elegibles otorgados hasta 2016.")
    partes=[]
    filas_fuente=0
    inicio=time.perf_counter()
    with pd.read_csv(RUTA_DATOS,usecols=cols_lectura,dtype=dtypes,chunksize=100_000,skipinitialspace=True,low_memory=False) as lector:
        for numero,bloque in enumerate(lector,start=1):
            filas_fuente+=len(bloque)
            fechas=pd.to_datetime(bloque["issue_d"],format="%b-%Y",errors="coerce")
            mask_fecha=fechas.notna()&(fechas.dt.year<=2016)
            bloque=bloque.loc[mask_fecha].copy()
            bloque["issue_d"]=fechas.loc[mask_fecha]
            bloque=bloque.loc[bloque["loan_status"].isin(MAPA_OBJETIVO)].copy()
            if bloque.empty: continue
            for col in ["earliest_cr_line","sec_app_earliest_cr_line"]:
                if col in bloque:
                    original=bloque[col].notna()
                    convertida=pd.to_datetime(bloque[col],format="%b-%Y",errors="coerce")
                    if (original&convertida.isna()).any(): raise ValueError(f"Fecha no reconocida: {col}")
                    bloque[col]=convertida
            if "term" in bloque:
                bloque["term"]=pd.to_numeric(bloque["term"].astype("string").str.strip().str.replace(" months","",regex=False),errors="raise").astype("float64")
            bloque["anio"]=bloque["issue_d"].dt.year.astype("int16")
            bloque["default"]=bloque["loan_status"].map(MAPA_OBJETIVO).astype("int8")
            bloque["fecha_prestamo"]=bloque["issue_d"]
            partes.append(bloque)
            if numero%10==0: print(f"  bloques recorridos {numero}; filas fuente {filas_fuente:,}",flush=True)
    datos=pd.concat(partes,ignore_index=True)
    del partes
    datos=datos.loc[datos.anio<=2016].copy()
    assert datos.anio.max()<=2016
    print(f"Préstamos en análisis: {len(datos):,}; años {datos.anio.min()}–{datos.anio.max()}; carga {time.perf_counter()-inicio:.1f} s")


    pct_2012=datos.loc[datos.anio<=2012,candidatas].isna().mean()*100
    variables_26=pct_2012[pct_2012<UMBRAL_FALTANTES].index.tolist()

    pct_2016=datos[candidatas].isna().mean()*100
    variables_59=pct_2016[pct_2016<UMBRAL_FALTANTES].index.tolist()
    if len(variables_26)!=26 or len(variables_59)!=59:
        raise ValueError(f"Se esperaban 26/59 variables; se obtuvieron {len(variables_26)}/{len(variables_59)}")

    lista_previa=(Path(referencias) if referencias is not None else raiz / "outputs/comparacion_variables_temporal")/"listas_26_y_59.csv"
    if lista_previa.exists():
        nombres_previos=pd.read_csv(lista_previa)["variable_59"].dropna().tolist()
        if set(nombres_previos)!=set(variables_59):
            raise AssertionError("La lista reconstruida de 59 variables no coincide con la anterior")
    print(f"Conjuntos: {len(variables_26)} fijas del primer entrenamiento; {len(variables_59)} exploratorias hasta 2016; {len(candidatas)} candidatas.")


    from src.preparacion_temporal import preparar_X as preparar_X



    PARAMETROS_MODELO={
        "regresion_logistica":{"C":0.13292918943162169,"class_weight":None,"solver":"lbfgs","max_iter":2000,"random_state":SEMILLA},
        "svm_lineal":{"C":0.010053940194693016,"class_weight":"balanced","dual":"auto","max_iter":5000,"random_state":SEMILLA},
    }

    def crear_modelo(nombre):
        if nombre=="regresion_logistica": return LogisticRegression(**PARAMETROS_MODELO[nombre])
        if nombre=="svm_lineal": return LinearSVC(**PARAMETROS_MODELO[nombre])
        raise ValueError(nombre)

    def entrenar_evaluar(nombre_modelo,anio_validacion,variables,con_indicadores,nombre_experimento):
        anio_final=anio_validacion-1
        mask_train=datos.anio<=anio_final
        mask_valid=datos.anio==anio_validacion
        Xtr=preparar_X(datos.loc[mask_train,variables],datos.loc[mask_train,"fecha_prestamo"])
        Xva=preparar_X(datos.loc[mask_valid,variables],datos.loc[mask_valid,"fecha_prestamo"])
        ytr=datos.loc[mask_train,"default"].to_numpy(dtype="int8")
        yva=datos.loc[mask_valid,"default"].to_numpy(dtype="int8")
        numericas=[col for col in Xtr if pd.api.types.is_numeric_dtype(Xtr[col])]
        categoricas=[col for col in Xtr if col not in numericas]
        flags=[]
        if con_indicadores:
            for col in numericas:
                if Xtr[col].isna().any():
                    flag=f"{col}__faltante"
                    Xtr[flag]=Xtr[col].isna().astype("int8")
                    Xva[flag]=Xva[col].isna().astype("int8")
                    flags.append(flag)
        transformers=[]
        if numericas:
            num_pipe=Pipeline([("imputar",SimpleImputer(strategy="median")),("escalar",StandardScaler(with_mean=False))])
            transformers.append(("numericas",num_pipe,numericas))
        if categoricas:
            cat_pipe=Pipeline([
                ("imputar",SimpleImputer(strategy="constant",fill_value="Sin información")),
                ("onehot",OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=True)),
            ])
            transformers.append(("categoricas",cat_pipe,categoricas))
        if flags:
            flag_pipe=Pipeline([("escalar",StandardScaler(with_mean=False))])
            transformers.append(("indicadores",flag_pipe,flags))
        pre=ColumnTransformer(transformers,remainder="drop",sparse_threshold=1.0)
        modelo=crear_modelo(nombre_modelo)
        pipeline=Pipeline([("preprocesar",pre),("modelo",modelo)])
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always",ConvergenceWarning)
            pipeline.fit(Xtr,ytr)
        segundos=time.perf_counter()-t
        if nombre_modelo=="regresion_logistica": puntuacion=pipeline.predict_proba(Xva)[:,1]
        else: puntuacion=pipeline.decision_function(Xva)
        resultado={
            "experimento":nombre_experimento,"modelo":nombre_modelo,
            "anio_final_entrenamiento":anio_final,"anio_validacion":anio_validacion,
            "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),
            "variables_originales":len(variables),"indicadores_faltantes":len(flags),
            "variables_numericas":len(numericas),"variables_categoricas":len(categoricas),
            "faltante_maximo_pct_entrenamiento":float(datos.loc[mask_train,variables].isna().mean().max()*100),
            "ROC_AUC":float(roc_auc_score(yva,puntuacion)),
            "PR_AUC":float(average_precision_score(yva,puntuacion)),
            "segundos_ajuste":float(segundos),
            "advertencia_convergencia":any(isinstance(w.message,ConvergenceWarning) for w in avisos),
        }
        pct=datos.loc[mask_train,variables].isna().mean()*100
        cobertura=pd.DataFrame({
            "experimento":nombre_experimento,"modelo":nombre_modelo,"anio_validacion":anio_validacion,
            "variable":variables,"faltante_pct_entrenamiento":[float(pct[v]) for v in variables],
            "indicador_creado":[bool(f"{({'earliest_cr_line':'meses_desde_earliest_cr_line','sec_app_earliest_cr_line':'meses_desde_sec_app_earliest_cr_line'}.get(v,v))}__faltante" in flags) for v in variables],
        })
        del pipeline,modelo,puntuacion,Xtr,Xva
        gc.collect()
        return resultado,cobertura

    metodos={"mediana":False,"mediana_indicadores":True}
    resultados_filas=[]
    coberturas=[]



    mask_train_corte=datos.anio<=2012
    pct_train=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_train[pct_train<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales: raise ValueError("No quedaron predictoras en el corte")
    conjuntos={"26_fijas":variables_26,"59_exploratorias":variables_59,"seleccion_temporal":variables_temporales}
    for conjunto,variables in conjuntos.items():

        for modelo in ["regresion_logistica","svm_lineal"]:
            for metodo,con_flags in metodos.items():
                experimento=f"{conjunto}_{modelo}_{metodo}"
                resultado,cobertura=entrenar_evaluar(modelo,2013,variables,con_flags,experimento)
                resultado["conjunto_variables"]=conjunto
                resultado["tipo_seleccion"]="exploratoria" if conjunto=="59_exploratorias" else "solo_entrenamiento"
                resultado["tratamiento"]="Mediana + indicadores" if con_flags else "Mediana"
                resultados_filas.append(resultado)
                coberturas.append(cobertura)
                print(f"{experimento}: {len(variables)} variables, {resultado['indicadores_faltantes']} indicadores; ROC={resultado['ROC_AUC']:.4f}, PR={resultado['PR_AUC']:.4f}",flush=True)
    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_lineales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2012, validación 2013.")



    mask_train_corte=datos.anio<=2013
    pct_train=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_train[pct_train<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales: raise ValueError("No quedaron predictoras en el corte")
    conjuntos={"26_fijas":variables_26,"59_exploratorias":variables_59,"seleccion_temporal":variables_temporales}
    for conjunto,variables in conjuntos.items():

        for modelo in ["regresion_logistica","svm_lineal"]:
            for metodo,con_flags in metodos.items():
                experimento=f"{conjunto}_{modelo}_{metodo}"
                resultado,cobertura=entrenar_evaluar(modelo,2014,variables,con_flags,experimento)
                resultado["conjunto_variables"]=conjunto
                resultado["tipo_seleccion"]="exploratoria" if conjunto=="59_exploratorias" else "solo_entrenamiento"
                resultado["tratamiento"]="Mediana + indicadores" if con_flags else "Mediana"
                resultados_filas.append(resultado)
                coberturas.append(cobertura)
                print(f"{experimento}: {len(variables)} variables, {resultado['indicadores_faltantes']} indicadores; ROC={resultado['ROC_AUC']:.4f}, PR={resultado['PR_AUC']:.4f}",flush=True)
    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_lineales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2013, validación 2014.")



    mask_train_corte=datos.anio<=2014
    pct_train=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_train[pct_train<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales: raise ValueError("No quedaron predictoras en el corte")
    conjuntos={"26_fijas":variables_26,"59_exploratorias":variables_59,"seleccion_temporal":variables_temporales}
    for conjunto,variables in conjuntos.items():

        for modelo in ["regresion_logistica","svm_lineal"]:
            for metodo,con_flags in metodos.items():
                experimento=f"{conjunto}_{modelo}_{metodo}"
                resultado,cobertura=entrenar_evaluar(modelo,2015,variables,con_flags,experimento)
                resultado["conjunto_variables"]=conjunto
                resultado["tipo_seleccion"]="exploratoria" if conjunto=="59_exploratorias" else "solo_entrenamiento"
                resultado["tratamiento"]="Mediana + indicadores" if con_flags else "Mediana"
                resultados_filas.append(resultado)
                coberturas.append(cobertura)
                print(f"{experimento}: {len(variables)} variables, {resultado['indicadores_faltantes']} indicadores; ROC={resultado['ROC_AUC']:.4f}, PR={resultado['PR_AUC']:.4f}",flush=True)
    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_lineales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2014, validación 2015.")



    mask_train_corte=datos.anio<=2015
    pct_train=datos.loc[mask_train_corte,candidatas].isna().mean()*100
    variables_temporales=pct_train[pct_train<UMBRAL_FALTANTES].index.tolist()
    if not variables_temporales: raise ValueError("No quedaron predictoras en el corte")
    conjuntos={"26_fijas":variables_26,"59_exploratorias":variables_59,"seleccion_temporal":variables_temporales}
    for conjunto,variables in conjuntos.items():

        for modelo in ["regresion_logistica","svm_lineal"]:
            for metodo,con_flags in metodos.items():
                experimento=f"{conjunto}_{modelo}_{metodo}"
                resultado,cobertura=entrenar_evaluar(modelo,2016,variables,con_flags,experimento)
                resultado["conjunto_variables"]=conjunto
                resultado["tipo_seleccion"]="exploratoria" if conjunto=="59_exploratorias" else "solo_entrenamiento"
                resultado["tratamiento"]="Mediana + indicadores" if con_flags else "Mediana"
                resultados_filas.append(resultado)
                coberturas.append(cobertura)
                print(f"{experimento}: {len(variables)} variables, {resultado['indicadores_faltantes']} indicadores; ROC={resultado['ROC_AUC']:.4f}, PR={resultado['PR_AUC']:.4f}",flush=True)
    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_lineales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2015, validación 2016.")


    resultados=pd.DataFrame(resultados_filas)
    cobertura_detalle=pd.concat(coberturas,ignore_index=True)
    resumen=(resultados.groupby(["conjunto_variables","tipo_seleccion","modelo","tratamiento"],as_index=False)
     .agg(ROC_AUC_promedio=("ROC_AUC","mean"),ROC_AUC_desviacion=("ROC_AUC","std"),
          PR_AUC_promedio=("PR_AUC","mean"),PR_AUC_desviacion=("PR_AUC","std"),
          variables_min=("variables_originales","min"),variables_max=("variables_originales","max"),
          indicadores_promedio=("indicadores_faltantes","mean"),cortes_convergencia=("advertencia_convergencia","sum")))
    resumen[["ROC_AUC_desviacion","PR_AUC_desviacion"]]=resumen[["ROC_AUC_desviacion","PR_AUC_desviacion"]].fillna(0)


    difs=[]
    for modelo in ["regresion_logistica","svm_lineal"]:
        baseline=resultados[(resultados.modelo==modelo)&(resultados.conjunto_variables=="26_fijas")&(resultados.tratamiento=="Mediana")]
        for conjunto in ["26_fijas","59_exploratorias","seleccion_temporal"]:
            for tratamiento in ["Mediana","Mediana + indicadores"]:
                actual=resultados[(resultados.modelo==modelo)&(resultados.conjunto_variables==conjunto)&(resultados.tratamiento==tratamiento)]
                if conjunto=="26_fijas" and tratamiento=="Mediana": continue
                m=baseline[["anio_validacion","ROC_AUC","PR_AUC"]].merge(actual[["anio_validacion","ROC_AUC","PR_AUC"]],on="anio_validacion",suffixes=("_base26_mediana","_actual"))
                difs.append({"modelo":modelo,"conjunto_variables":conjunto,"tratamiento":tratamiento,
                    "delta_ROC_AUC_vs_26_mediana":float((m.ROC_AUC_actual-m.ROC_AUC_base26_mediana).mean()),
                    "cortes_ROC_mayor_base":int((m.ROC_AUC_actual>m.ROC_AUC_base26_mediana).sum()),
                    "delta_PR_AUC_vs_26_mediana":float((m.PR_AUC_actual-m.PR_AUC_base26_mediana).mean()),
                    "cortes_PR_mayor_base":int((m.PR_AUC_actual>m.PR_AUC_base26_mediana).sum())})
    diferencias=pd.DataFrame(difs)


    comparacion_ind=[]
    for modelo in ["regresion_logistica","svm_lineal"]:
        for conjunto in ["26_fijas","59_exploratorias","seleccion_temporal"]:
            sin=resultados[(resultados.modelo==modelo)&(resultados.conjunto_variables==conjunto)&(resultados.tratamiento=="Mediana")]
            con=resultados[(resultados.modelo==modelo)&(resultados.conjunto_variables==conjunto)&(resultados.tratamiento=="Mediana + indicadores")]
            m=sin[["anio_validacion","ROC_AUC","PR_AUC"]].merge(con[["anio_validacion","ROC_AUC","PR_AUC"]],on="anio_validacion",suffixes=("_sin","_con"))
            comparacion_ind.append({"modelo":modelo,"conjunto_variables":conjunto,
              "delta_ROC_AUC_indicadores":float((m.ROC_AUC_con-m.ROC_AUC_sin).mean()),
              "cortes_ROC_mejora":int((m.ROC_AUC_con>m.ROC_AUC_sin).sum()),
              "delta_PR_AUC_indicadores":float((m.PR_AUC_con-m.PR_AUC_sin).mean()),
              "cortes_PR_mejora":int((m.PR_AUC_con>m.PR_AUC_sin).sum())})
    indicadores_resumen=pd.DataFrame(comparacion_ind)

    display(resumen.round(5))
    display(diferencias.round(5))
    display(indicadores_resumen.round(5))
    display(resultados[["modelo","conjunto_variables","tratamiento","anio_validacion","variables_originales","indicadores_faltantes","ROC_AUC","PR_AUC","advertencia_convergencia"]].sort_values(["modelo","conjunto_variables","tratamiento","anio_validacion"]).round(5))


    resumen_cobertura=(cobertura_detalle.groupby(["modelo","experimento","anio_validacion"],as_index=False)
     .agg(variables=("variable","count"),variables_con_faltantes=("faltante_pct_entrenamiento",lambda s:int((s>0).sum())),
          faltante_promedio_pct=("faltante_pct_entrenamiento","mean"),faltante_maximo_pct=("faltante_pct_entrenamiento","max"),
          indicadores=("indicador_creado","sum")))
    display(resumen_cobertura.round(2))


    for modelo in ["regresion_logistica","svm_lineal"]:
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        subset=resultados[resultados.modelo==modelo]
        for conj in ["26_fijas","59_exploratorias","seleccion_temporal"]:
            for trat in ["Mediana","Mediana + indicadores"]:
                s=subset[(subset.conjunto_variables==conj)&(subset.tratamiento==trat)].sort_values("anio_validacion")
                label=f"{conj} / {trat}"
                axes[0].plot(s.anio_validacion,s.ROC_AUC,marker="o",label=label)
                axes[1].plot(s.anio_validacion,s.PR_AUC,marker="o",label=label)
        axes[0].set_title(f"{modelo}: ROC-AUC por año")
        axes[1].set_title(f"{modelo}: PR-AUC por año")
        for ax in axes:
            ax.set_xticks(ANIOS_VALIDACION); ax.set_xlabel("Año de validación"); ax.grid(alpha=.25)
        axes[0].legend(fontsize=8,bbox_to_anchor=(1.02,1),loc="upper left")
        fig.tight_layout(); fig.savefig(SALIDAS / f"figura_{fig.number}.png", dpi=150, bbox_inches="tight"); display(fig); plt.close(fig)

    salida=SALIDAS/"comparacion_lineales"
    salida.mkdir(parents=True,exist_ok=True)
    resultados.to_csv(salida/"metricas_por_configuracion_y_anio.csv",index=False)
    resumen.to_csv(salida/"resumen_promedios_y_variabilidad.csv",index=False)
    diferencias.to_csv(salida/"diferencias_vs_baseline_26_mediana.csv",index=False)
    indicadores_resumen.to_csv(salida/"diferencias_indicadores_vs_mediana.csv",index=False)
    cobertura_detalle.to_csv(salida/"faltantes_por_variable_y_corte.csv",index=False)
    resumen_cobertura.to_csv(salida/"resumen_cobertura_por_corte.csv",index=False)
    pd.DataFrame({"variables_26":pd.Series(variables_26),"variables_59":pd.Series(variables_59),"candidatas":pd.Series(candidatas)}).to_csv(salida/"listas_variables.csv",index=False)
    print("Resultados guardados en:",salida)

    for _,r in diferencias.iterrows():
        print(f"{r['modelo']} / {r['conjunto_variables']} / {r['tratamiento']} vs 26 mediana: ΔROC={r['delta_ROC_AUC_vs_26_mediana']:+.5f} en {int(r['cortes_ROC_mayor_base'])}/4; ΔPR={r['delta_PR_AUC_vs_26_mediana']:+.5f} en {int(r['cortes_PR_mayor_base'])}/4.")

    return {k: v for k, v in locals().items() if k in ("resumen", "resultados", "resumen_modelos", "resultados_por_corte")}
