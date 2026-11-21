"""Experimento oficial: comparacion_umbrales. Conserva semillas, cortes y parámetros.
Las nuevas ejecuciones escriben en una carpeta independiente.
"""
from pathlib import Path

def ejecutar(raiz, salidas, ruta_datos=None):
    from pathlib import Path
    raiz = Path(raiz).resolve()
    SALIDAS = Path(salidas).resolve()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import sys,time,gc,warnings
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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score,average_precision_score
    from sklearn.exceptions import ConvergenceWarning

    RAIZ = raiz
    if str(RAIZ) not in sys.path: sys.path.insert(0,str(RAIZ))
    from src.config import localizar_datos,MAPA_OBJETIVO,SEMILLA
    from src.modelado import TIPOS_AUDITADOS
    from src import features as ft

    UMBRALES=[10,30,50,75]
    ANIOS_VALIDACION=[2013,2014,2015,2016]
    RUTA=localizar_datos(ruta_datos)
    columnas_fuente=pd.read_csv(RUTA,nrows=0).columns.tolist()
    candidatas=ft.seleccionar_predictoras(pd.DataFrame(columns=columnas_fuente),[],incluir_precio_lc=False)
    cols_lectura=candidatas+["loan_status","issue_d"]
    dtypes={col:("string" if TIPOS_AUDITADOS[col] in ["category","string"] else TIPOS_AUDITADOS[col]) for col in cols_lectura}
    print(f"Candidatas prefiltrado: {len(candidatas)}")
    print("Lectura por bloques; se retienen préstamos elegibles con fecha hasta 2016...")
    partes=[]; filas_leidas=0; t0=time.perf_counter()
    with pd.read_csv(RUTA,usecols=cols_lectura,dtype=dtypes,chunksize=100_000,skipinitialspace=True,low_memory=False) as lector:
        for numero,bloque in enumerate(lector,start=1):
            filas_leidas+=len(bloque)
            fechas=pd.to_datetime(bloque["issue_d"],format="%b-%Y",errors="coerce")
            mask_fecha=fechas.notna()&(fechas.dt.year<=2016)
            bloque=bloque.loc[mask_fecha].copy(); bloque["issue_d"]=fechas.loc[mask_fecha]

            bloque=bloque.loc[bloque["loan_status"].isin(MAPA_OBJETIVO)].copy()
            if bloque.empty: continue
            for col in ["earliest_cr_line","sec_app_earliest_cr_line"]:
                if col in bloque:
                    original=bloque[col].notna()
                    convertido=pd.to_datetime(bloque[col],format="%b-%Y",errors="coerce")
                    if (original&convertido.isna()).any(): raise ValueError(f"Fecha no reconocida en {col}")
                    bloque[col]=convertido
            if "term" in bloque:
                bloque["term"]=pd.to_numeric(bloque["term"].astype("string").str.strip().str.replace(" months","",regex=False),errors="raise").astype("float64")
            bloque["anio"]=bloque["issue_d"].dt.year.astype("int16")
            bloque["default"]=bloque["loan_status"].map(MAPA_OBJETIVO).astype("int8")
            bloque["fecha_prestamo"]=bloque["issue_d"]
            partes.append(bloque)
            if numero%10==0: print(f"  bloques {numero}; filas fuente recorridas {filas_leidas:,}",flush=True)
    datos=pd.concat(partes,ignore_index=True); del partes
    datos=datos.loc[datos.anio<=2016].copy(); assert datos.anio.max()<=2016
    print(f"Datos: {len(datos):,} préstamos; años {datos.anio.min()}–{datos.anio.max()}; carga {time.perf_counter()-t0:.1f} s")


    filas_exclusion=[]
    for nombre,lista,motivo in [
        ("Identificadores",ft.IDENTIFICADORES,"No identifican capacidad de pago"),
        ("Texto libre",ft.TEXTO_LIBRE,"Fuera del alcance de este análisis"),
        ("Alta cardinalidad / código",ft.DESCARTADAS_EXTRA,"Excluido según reglas actuales del proyecto"),
        ("Precio / calificación LC",ft.PRECIO_LC,"No se incluye en este escenario predictivo"),
        ("Información posterior",ft.columnas_posteriores(columnas_fuente),"Generada después del otorgamiento; leakage"),
    ]:
        for col in lista:
            if col in columnas_fuente: filas_exclusion.append({"variable":col,"grupo":nombre,"motivo":motivo})
    tabla_exclusiones=pd.DataFrame(filas_exclusion).drop_duplicates("variable")
    print("Exclusiones explícitas por disponibilidad/alcance:",len(tabla_exclusiones))
    display(tabla_exclusiones)
    print("Candidatas disponibles al otorgamiento:",len(candidatas))


    from src.preparacion_temporal import preparar_X as preparar_X



    PARAMETROS={
     "regresion_logistica":{"C":0.13292918943162169,"class_weight":None,"solver":"lbfgs","max_iter":2000,"random_state":SEMILLA},
     "svm_lineal":{"C":0.010053940194693016,"class_weight":"balanced","dual":"auto","max_iter":5000,"random_state":SEMILLA},
     "gradient_boosting":{"learning_rate":0.08118441500491742,"max_iter":182,"max_leaf_nodes":31,"min_samples_leaf":46,"l2_regularization":0.32877474139911184,"early_stopping":False,"random_state":SEMILLA},
    }

    def modelo_nuevo(nombre):
        if nombre=="regresion_logistica": return LogisticRegression(**PARAMETROS[nombre])
        if nombre=="svm_lineal": return LinearSVC(**PARAMETROS[nombre])
        if nombre=="gradient_boosting": return HistGradientBoostingClassifier(**PARAMETROS[nombre])
        raise ValueError(nombre)

    def construir_preprocesador(X_train,denso=False):
        numericas=[c for c in X_train if pd.api.types.is_numeric_dtype(X_train[c])]
        categoricas=[c for c in X_train if c not in numericas]
        transformadores=[]
        if numericas:
            transformadores.append(("numericas",Pipeline([
                ("imputar",SimpleImputer(strategy="median")),
                ("escalar",StandardScaler(with_mean=denso)),
            ]),numericas))
        if categoricas:
            transformadores.append(("categoricas",Pipeline([
                ("imputar",SimpleImputer(strategy="constant",fill_value="Sin información")),
                ("onehot",OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=not denso,dtype=np.float32)),
            ]),categoricas))
        return ColumnTransformer(transformadores,remainder="drop",sparse_threshold=0.0 if denso else 1.0),numericas,categoricas



    resultados_filas=[]
    selecciones_filas=[]


    anio=2013
    mask_train=datos.anio<=2012
    mask_valid=datos.anio==2013
    pct_train=datos.loc[mask_train,candidatas].isna().mean()*100
    for umbral in UMBRALES:
        filas=[]
        seleccion=pct_train[pct_train<=umbral].index.tolist()
        excluidas=pct_train[pct_train>umbral].index.tolist()
        if not seleccion: raise ValueError(f"No hay variables bajo el umbral {umbral} en {anio}")
        for variable in candidatas:
            filas.append({"anio_validacion":2013,"anio_final_entrenamiento":2012,"umbral_pct":umbral,
                          "variable":variable,"faltante_pct_entrenamiento":float(pct_train[variable]),
                          "seleccionada":variable in seleccion,
                          "decision":"conservar" if variable in seleccion else "excluir"})
        selecciones_filas.extend(filas)

        Xtr=preparar_X(datos.loc[mask_train,seleccion],datos.loc[mask_train,"fecha_prestamo"])
        Xva=preparar_X(datos.loc[mask_valid,seleccion],datos.loc[mask_valid,"fecha_prestamo"])
        ytr=datos.loc[mask_train,"default"].to_numpy(dtype="int8")
        yva=datos.loc[mask_valid,"default"].to_numpy(dtype="int8")
        if len(np.unique(yva))<2: raise ValueError(f"Falta una clase en validación 2013")


        pre_lineal,numericas,categoricas=construir_preprocesador(Xtr,denso=False)
        Xtr_lineal=pre_lineal.fit_transform(Xtr); Xva_lineal=pre_lineal.transform(Xva)
        if not sparse.issparse(Xtr_lineal):
            Xtr_lineal=sparse.csr_matrix(Xtr_lineal); Xva_lineal=sparse.csr_matrix(Xva_lineal)
        for nombre_modelo in ["regresion_logistica","svm_lineal"]:
            modelo=modelo_nuevo(nombre_modelo)
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always",ConvergenceWarning)
                modelo.fit(Xtr_lineal,ytr)
            segundos=time.perf_counter()-t
            puntuacion=(modelo.predict_proba(Xva_lineal)[:,1] if nombre_modelo=="regresion_logistica"
                        else modelo.decision_function(Xva_lineal))
            resultados_fila={"modelo":nombre_modelo,"umbral_pct":umbral,"anio_validacion":2013,
                "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
                "columnas_transformadas":int(Xtr_lineal.shape[1]),"variables_numericas":len(numericas),
                "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
                "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
                "advertencia_convergencia":any(isinstance(w.message,ConvergenceWarning) for w in avisos)}
            resultados_filas.append(resultados_fila)
            print(f"{nombre_modelo} | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
            del modelo,puntuacion
            gc.collect()


        pre_gb,_,_=construir_preprocesador(Xtr,denso=True)
        Xtr_gb=pre_gb.fit_transform(Xtr); Xva_gb=pre_gb.transform(Xva)
        modelo=modelo_nuevo("gradient_boosting")
        t=time.perf_counter(); modelo.fit(Xtr_gb,ytr); segundos=time.perf_counter()-t
        puntuacion=modelo.predict_proba(Xva_gb)[:,1]
        resultados_fila={"modelo":"gradient_boosting","umbral_pct":umbral,"anio_validacion":2013,
            "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
            "columnas_transformadas":int(Xtr_gb.shape[1]),"variables_numericas":len(numericas),
            "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
            "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
            "advertencia_convergencia":False}
        resultados_filas.append(resultados_fila)
        print(f"gradient_boosting | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
        del pre_lineal,Xtr_lineal,Xva_lineal,pre_gb,Xtr_gb,Xva_gb,modelo,puntuacion,Xtr,Xva,ytr,yva
        gc.collect()

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_umbrales_parcial.csv",index=False)
    pd.DataFrame(selecciones_filas).to_csv(SALIDAS/"selecciones_umbrales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2012, validación 2013.")



    anio=2014
    mask_train=datos.anio<=2013
    mask_valid=datos.anio==2014
    pct_train=datos.loc[mask_train,candidatas].isna().mean()*100
    for umbral in UMBRALES:
        filas=[]
        seleccion=pct_train[pct_train<=umbral].index.tolist()
        excluidas=pct_train[pct_train>umbral].index.tolist()
        if not seleccion: raise ValueError(f"No hay variables bajo el umbral {umbral} en {anio}")
        for variable in candidatas:
            filas.append({"anio_validacion":2014,"anio_final_entrenamiento":2013,"umbral_pct":umbral,
                          "variable":variable,"faltante_pct_entrenamiento":float(pct_train[variable]),
                          "seleccionada":variable in seleccion,
                          "decision":"conservar" if variable in seleccion else "excluir"})
        selecciones_filas.extend(filas)

        Xtr=preparar_X(datos.loc[mask_train,seleccion],datos.loc[mask_train,"fecha_prestamo"])
        Xva=preparar_X(datos.loc[mask_valid,seleccion],datos.loc[mask_valid,"fecha_prestamo"])
        ytr=datos.loc[mask_train,"default"].to_numpy(dtype="int8")
        yva=datos.loc[mask_valid,"default"].to_numpy(dtype="int8")
        if len(np.unique(yva))<2: raise ValueError(f"Falta una clase en validación 2014")


        pre_lineal,numericas,categoricas=construir_preprocesador(Xtr,denso=False)
        Xtr_lineal=pre_lineal.fit_transform(Xtr); Xva_lineal=pre_lineal.transform(Xva)
        if not sparse.issparse(Xtr_lineal):
            Xtr_lineal=sparse.csr_matrix(Xtr_lineal); Xva_lineal=sparse.csr_matrix(Xva_lineal)
        for nombre_modelo in ["regresion_logistica","svm_lineal"]:
            modelo=modelo_nuevo(nombre_modelo)
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always",ConvergenceWarning)
                modelo.fit(Xtr_lineal,ytr)
            segundos=time.perf_counter()-t
            puntuacion=(modelo.predict_proba(Xva_lineal)[:,1] if nombre_modelo=="regresion_logistica"
                        else modelo.decision_function(Xva_lineal))
            resultados_fila={"modelo":nombre_modelo,"umbral_pct":umbral,"anio_validacion":2014,
                "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
                "columnas_transformadas":int(Xtr_lineal.shape[1]),"variables_numericas":len(numericas),
                "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
                "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
                "advertencia_convergencia":any(isinstance(w.message,ConvergenceWarning) for w in avisos)}
            resultados_filas.append(resultados_fila)
            print(f"{nombre_modelo} | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
            del modelo,puntuacion
            gc.collect()


        pre_gb,_,_=construir_preprocesador(Xtr,denso=True)
        Xtr_gb=pre_gb.fit_transform(Xtr); Xva_gb=pre_gb.transform(Xva)
        modelo=modelo_nuevo("gradient_boosting")
        t=time.perf_counter(); modelo.fit(Xtr_gb,ytr); segundos=time.perf_counter()-t
        puntuacion=modelo.predict_proba(Xva_gb)[:,1]
        resultados_fila={"modelo":"gradient_boosting","umbral_pct":umbral,"anio_validacion":2014,
            "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
            "columnas_transformadas":int(Xtr_gb.shape[1]),"variables_numericas":len(numericas),
            "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
            "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
            "advertencia_convergencia":False}
        resultados_filas.append(resultados_fila)
        print(f"gradient_boosting | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
        del pre_lineal,Xtr_lineal,Xva_lineal,pre_gb,Xtr_gb,Xva_gb,modelo,puntuacion,Xtr,Xva,ytr,yva
        gc.collect()

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_umbrales_parcial.csv",index=False)
    pd.DataFrame(selecciones_filas).to_csv(SALIDAS/"selecciones_umbrales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2013, validación 2014.")



    anio=2015
    mask_train=datos.anio<=2014
    mask_valid=datos.anio==2015
    pct_train=datos.loc[mask_train,candidatas].isna().mean()*100
    for umbral in UMBRALES:
        filas=[]
        seleccion=pct_train[pct_train<=umbral].index.tolist()
        excluidas=pct_train[pct_train>umbral].index.tolist()
        if not seleccion: raise ValueError(f"No hay variables bajo el umbral {umbral} en {anio}")
        for variable in candidatas:
            filas.append({"anio_validacion":2015,"anio_final_entrenamiento":2014,"umbral_pct":umbral,
                          "variable":variable,"faltante_pct_entrenamiento":float(pct_train[variable]),
                          "seleccionada":variable in seleccion,
                          "decision":"conservar" if variable in seleccion else "excluir"})
        selecciones_filas.extend(filas)

        Xtr=preparar_X(datos.loc[mask_train,seleccion],datos.loc[mask_train,"fecha_prestamo"])
        Xva=preparar_X(datos.loc[mask_valid,seleccion],datos.loc[mask_valid,"fecha_prestamo"])
        ytr=datos.loc[mask_train,"default"].to_numpy(dtype="int8")
        yva=datos.loc[mask_valid,"default"].to_numpy(dtype="int8")
        if len(np.unique(yva))<2: raise ValueError(f"Falta una clase en validación 2015")


        pre_lineal,numericas,categoricas=construir_preprocesador(Xtr,denso=False)
        Xtr_lineal=pre_lineal.fit_transform(Xtr); Xva_lineal=pre_lineal.transform(Xva)
        if not sparse.issparse(Xtr_lineal):
            Xtr_lineal=sparse.csr_matrix(Xtr_lineal); Xva_lineal=sparse.csr_matrix(Xva_lineal)
        for nombre_modelo in ["regresion_logistica","svm_lineal"]:
            modelo=modelo_nuevo(nombre_modelo)
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always",ConvergenceWarning)
                modelo.fit(Xtr_lineal,ytr)
            segundos=time.perf_counter()-t
            puntuacion=(modelo.predict_proba(Xva_lineal)[:,1] if nombre_modelo=="regresion_logistica"
                        else modelo.decision_function(Xva_lineal))
            resultados_fila={"modelo":nombre_modelo,"umbral_pct":umbral,"anio_validacion":2015,
                "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
                "columnas_transformadas":int(Xtr_lineal.shape[1]),"variables_numericas":len(numericas),
                "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
                "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
                "advertencia_convergencia":any(isinstance(w.message,ConvergenceWarning) for w in avisos)}
            resultados_filas.append(resultados_fila)
            print(f"{nombre_modelo} | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
            del modelo,puntuacion
            gc.collect()


        pre_gb,_,_=construir_preprocesador(Xtr,denso=True)
        Xtr_gb=pre_gb.fit_transform(Xtr); Xva_gb=pre_gb.transform(Xva)
        modelo=modelo_nuevo("gradient_boosting")
        t=time.perf_counter(); modelo.fit(Xtr_gb,ytr); segundos=time.perf_counter()-t
        puntuacion=modelo.predict_proba(Xva_gb)[:,1]
        resultados_fila={"modelo":"gradient_boosting","umbral_pct":umbral,"anio_validacion":2015,
            "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
            "columnas_transformadas":int(Xtr_gb.shape[1]),"variables_numericas":len(numericas),
            "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
            "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
            "advertencia_convergencia":False}
        resultados_filas.append(resultados_fila)
        print(f"gradient_boosting | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
        del pre_lineal,Xtr_lineal,Xva_lineal,pre_gb,Xtr_gb,Xva_gb,modelo,puntuacion,Xtr,Xva,ytr,yva
        gc.collect()

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_umbrales_parcial.csv",index=False)
    pd.DataFrame(selecciones_filas).to_csv(SALIDAS/"selecciones_umbrales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2014, validación 2015.")



    anio=2016
    mask_train=datos.anio<=2015
    mask_valid=datos.anio==2016
    pct_train=datos.loc[mask_train,candidatas].isna().mean()*100
    for umbral in UMBRALES:
        filas=[]
        seleccion=pct_train[pct_train<=umbral].index.tolist()
        excluidas=pct_train[pct_train>umbral].index.tolist()
        if not seleccion: raise ValueError(f"No hay variables bajo el umbral {umbral} en {anio}")
        for variable in candidatas:
            filas.append({"anio_validacion":2016,"anio_final_entrenamiento":2015,"umbral_pct":umbral,
                          "variable":variable,"faltante_pct_entrenamiento":float(pct_train[variable]),
                          "seleccionada":variable in seleccion,
                          "decision":"conservar" if variable in seleccion else "excluir"})
        selecciones_filas.extend(filas)

        Xtr=preparar_X(datos.loc[mask_train,seleccion],datos.loc[mask_train,"fecha_prestamo"])
        Xva=preparar_X(datos.loc[mask_valid,seleccion],datos.loc[mask_valid,"fecha_prestamo"])
        ytr=datos.loc[mask_train,"default"].to_numpy(dtype="int8")
        yva=datos.loc[mask_valid,"default"].to_numpy(dtype="int8")
        if len(np.unique(yva))<2: raise ValueError(f"Falta una clase en validación 2016")


        pre_lineal,numericas,categoricas=construir_preprocesador(Xtr,denso=False)
        Xtr_lineal=pre_lineal.fit_transform(Xtr); Xva_lineal=pre_lineal.transform(Xva)
        if not sparse.issparse(Xtr_lineal):
            Xtr_lineal=sparse.csr_matrix(Xtr_lineal); Xva_lineal=sparse.csr_matrix(Xva_lineal)
        for nombre_modelo in ["regresion_logistica","svm_lineal"]:
            modelo=modelo_nuevo(nombre_modelo)
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always",ConvergenceWarning)
                modelo.fit(Xtr_lineal,ytr)
            segundos=time.perf_counter()-t
            puntuacion=(modelo.predict_proba(Xva_lineal)[:,1] if nombre_modelo=="regresion_logistica"
                        else modelo.decision_function(Xva_lineal))
            resultados_fila={"modelo":nombre_modelo,"umbral_pct":umbral,"anio_validacion":2016,
                "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
                "columnas_transformadas":int(Xtr_lineal.shape[1]),"variables_numericas":len(numericas),
                "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
                "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
                "advertencia_convergencia":any(isinstance(w.message,ConvergenceWarning) for w in avisos)}
            resultados_filas.append(resultados_fila)
            print(f"{nombre_modelo} | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
            del modelo,puntuacion
            gc.collect()


        pre_gb,_,_=construir_preprocesador(Xtr,denso=True)
        Xtr_gb=pre_gb.fit_transform(Xtr); Xva_gb=pre_gb.transform(Xva)
        modelo=modelo_nuevo("gradient_boosting")
        t=time.perf_counter(); modelo.fit(Xtr_gb,ytr); segundos=time.perf_counter()-t
        puntuacion=modelo.predict_proba(Xva_gb)[:,1]
        resultados_fila={"modelo":"gradient_boosting","umbral_pct":umbral,"anio_validacion":2016,
            "filas_entrenamiento":len(ytr),"filas_validacion":len(yva),"variables_originales":len(seleccion),
            "columnas_transformadas":int(Xtr_gb.shape[1]),"variables_numericas":len(numericas),
            "variables_categoricas":len(categoricas),"ROC_AUC":float(roc_auc_score(yva,puntuacion)),
            "PR_AUC":float(average_precision_score(yva,puntuacion)),"segundos_ajuste":float(segundos),
            "advertencia_convergencia":False}
        resultados_filas.append(resultados_fila)
        print(f"gradient_boosting | ≤{umbral}% | {anio}: {len(seleccion)} vars, ROC={resultados_fila['ROC_AUC']:.4f}, PR={resultados_fila['PR_AUC']:.4f}",flush=True)
        del pre_lineal,Xtr_lineal,Xva_lineal,pre_gb,Xtr_gb,Xva_gb,modelo,puntuacion,Xtr,Xva,ytr,yva
        gc.collect()

    pd.DataFrame(resultados_filas).to_csv(SALIDAS/"comparacion_umbrales_parcial.csv",index=False)
    pd.DataFrame(selecciones_filas).to_csv(SALIDAS/"selecciones_umbrales_parcial.csv",index=False)
    print("Corte terminado: train hasta 2015, validación 2016.")




    resultados=pd.read_csv(SALIDAS/"comparacion_umbrales_parcial.csv")
    selecciones=pd.read_csv(SALIDAS/"selecciones_umbrales_parcial.csv")
    selecciones=selecciones.drop_duplicates(["anio_validacion","umbral_pct","variable"],keep="first").reset_index(drop=True)
    resumen=(resultados.groupby(["modelo","umbral_pct"],as_index=False)
     .agg(variables_promedio=("variables_originales","mean"),variables_min=("variables_originales","min"),variables_max=("variables_originales","max"),
          ROC_AUC_promedio=("ROC_AUC","mean"),ROC_AUC_desviacion=("ROC_AUC","std"),
          PR_AUC_promedio=("PR_AUC","mean"),PR_AUC_desviacion=("PR_AUC","std"),
          columnas_transformadas_promedio=("columnas_transformadas","mean"),cortes_convergencia=("advertencia_convergencia","sum")))
    resumen[["ROC_AUC_desviacion","PR_AUC_desviacion"]]=resumen[["ROC_AUC_desviacion","PR_AUC_desviacion"]].fillna(0)


    referencias=pd.DataFrame([
     {"modelo":"regresion_logistica","variables":"26 fijas","ROC_AUC":0.6927,"tipo":"referencia anterior"},
     {"modelo":"regresion_logistica","variables":"59 exploratorias","ROC_AUC":0.7054,"tipo":"exploratoria, selección hasta 2016"},
     {"modelo":"svm_lineal","variables":"26 fijas","ROC_AUC":0.6926,"tipo":"referencia anterior"},
     {"modelo":"svm_lineal","variables":"59 exploratorias","ROC_AUC":0.7052,"tipo":"exploratoria, selección hasta 2016"},
     {"modelo":"gradient_boosting","variables":"26 fijas","ROC_AUC":0.7000,"tipo":"referencia anterior"},
     {"modelo":"gradient_boosting","variables":"59 exploratorias","ROC_AUC":0.7131,"tipo":"exploratoria, selección hasta 2016"},
    ])
    display(resumen.round(5))
    display(referencias)

    comparacion_referencias=[]
    for _,fila in resumen.iterrows():
        refs=referencias[referencias.modelo==fila.modelo]
        for _,ref in refs.iterrows():
            comparacion_referencias.append({
                "modelo":fila.modelo,"umbral_pct":int(fila.umbral_pct),
                "variables_promedio":float(fila.variables_promedio),
                "ROC_AUC_umbral":float(fila.ROC_AUC_promedio),
                "referencia":ref["variables"],"tipo_referencia":ref["tipo"],
                "ROC_AUC_referencia":float(ref["ROC_AUC"]),
                "diferencia_ROC_AUC":float(fila.ROC_AUC_promedio-ref["ROC_AUC"]),
            })
    comparacion_referencias=pd.DataFrame(comparacion_referencias)
    display(comparacion_referencias.round(5))


    display(resultados.sort_values(["modelo","umbral_pct","anio_validacion"])[["modelo","umbral_pct","anio_validacion","variables_originales","ROC_AUC","PR_AUC","advertencia_convergencia"]].round(5))
    conteos=(selecciones.groupby(["anio_validacion","umbral_pct","decision"],as_index=False).size()
             .pivot(index=["anio_validacion","umbral_pct"],columns="decision",values="size").fillna(0).reset_index())
    display(conteos)

    salida=SALIDAS/"comparacion_umbrales_faltantes"
    salida.mkdir(parents=True,exist_ok=True)


    fig,ax=plt.subplots(figsize=(8,5))
    for modelo in ["regresion_logistica","svm_lineal","gradient_boosting"]:
     s=resumen[resumen.modelo==modelo].sort_values("umbral_pct")
     ax.plot(s.umbral_pct,s.ROC_AUC_promedio,marker="o",label=modelo)
    ax.set(title="ROC-AUC promedio frente al umbral",xlabel="Máximo de faltantes permitido (%)",ylabel="ROC-AUC")
    ax.set_xticks(UMBRALES); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(salida/"01_roc_auc_promedio.png",dpi=160,bbox_inches="tight"); display(fig); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    for modelo in ["regresion_logistica","svm_lineal","gradient_boosting"]:
     s=resumen[resumen.modelo==modelo].sort_values("umbral_pct")
     ax.plot(s.umbral_pct,s.PR_AUC_promedio,marker="o",label=modelo)
    ax.set(title="PR-AUC promedio frente al umbral",xlabel="Máximo de faltantes permitido (%)",ylabel="PR-AUC (Average Precision)")
    ax.set_xticks(UMBRALES); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(salida/"02_pr_auc_promedio.png",dpi=160,bbox_inches="tight"); display(fig); plt.close(fig)

    fig,axes=plt.subplots(1,4,figsize=(16,4),sharey=True)
    for ax,anio in zip(axes,ANIOS_VALIDACION):
     sub=conteos[conteos.anio_validacion==anio].sort_values("umbral_pct")
     ax.plot(sub.umbral_pct,sub.get("conservar",0),marker="o",label="Conservar")
     ax.set_title(str(anio)); ax.set_xticks(UMBRALES); ax.grid(alpha=.25); ax.set_xlabel("Umbral (%)")
    axes[0].set_ylabel("Número de variables seleccionadas"); fig.suptitle("Variables seleccionadas por corte y umbral"); fig.tight_layout(); fig.savefig(salida/"03_variables_por_corte.png",dpi=160,bbox_inches="tight"); display(fig); plt.close(fig)

    for metrica,titulo in [("ROC_AUC","ROC-AUC por año"),("PR_AUC","PR-AUC por año")]:
     fig,axes=plt.subplots(1,3,figsize=(16,4),sharey=False)
     for ax,modelo in zip(axes,["regresion_logistica","svm_lineal","gradient_boosting"]):
      sub=resultados[resultados.modelo==modelo]
      for umbral in UMBRALES:
       s=sub[sub.umbral_pct==umbral].sort_values("anio_validacion")
       ax.plot(s.anio_validacion,s[metrica],marker="o",label=f"≤{umbral}%")
      ax.set_title(modelo); ax.set_xticks(ANIOS_VALIDACION); ax.set_xlabel("Año")
      ax.grid(alpha=.25); ax.legend(fontsize=8)
     axes[0].set_ylabel(metrica); fig.suptitle(titulo); fig.tight_layout()
     nombre_archivo="04_roc_auc_por_anio.png" if metrica=="ROC_AUC" else "05_pr_auc_por_anio.png"
     fig.savefig(salida/nombre_archivo,dpi=160,bbox_inches="tight"); display(fig); plt.close(fig)

    salida=SALIDAS/"comparacion_umbrales_faltantes"
    salida.mkdir(parents=True,exist_ok=True)
    resultados.to_csv(salida/"metricas_por_modelo_umbral_corte.csv",index=False)
    resumen.to_csv(salida/"resumen_promedios_y_variabilidad.csv",index=False)
    selecciones.to_csv(salida/"variables_seleccionadas_y_excluidas.csv",index=False)
    conteos.to_csv(salida/"conteo_variables_por_corte_y_umbral.csv",index=False)
    referencias.to_csv(salida/"referencias_experimentos_anteriores.csv",index=False)
    comparacion_referencias.to_csv(salida/"comparacion_con_referencias_anteriores.csv",index=False)
    tabla_exclusiones.to_csv(salida/"exclusiones_por_disponibilidad_y_alcance.csv",index=False)
    print("Resultados guardados en:",salida)


    for modelo in ["regresion_logistica","svm_lineal","gradient_boosting"]:
     s=resumen[resumen.modelo==modelo].sort_values("ROC_AUC_promedio",ascending=False)
     top=s.iloc[0]
     print(f"{modelo}: umbral con mayor ROC-AUC promedio ≤{int(top.umbral_pct)}% (ROC={top.ROC_AUC_promedio:.5f}, PR={top.PR_AUC_promedio:.5f}, vars promedio={top.variables_promedio:.1f}, desviación ROC={top.ROC_AUC_desviacion:.5f}).")

    return {k: v for k, v in locals().items() if k in ("resumen", "resultados", "resumen_modelos", "resultados_por_corte")}
