from pathlib import Path

def ejecutar(raiz, salidas):
    from pathlib import Path
    raiz = Path(raiz).resolve()
    SALIDAS = Path(salidas).resolve()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    import sys
    import time
    import platform
    import numpy as np
    import pandas as pd
    from IPython.display import display

    RAIZ = raiz
    assert (RAIZ / "src").exists(), "Ejecutar desde el proyecto o desde notebooks."
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    from src.data import inferir_tipos, COLUMNAS_FECHA
    from src.config import localizar_datos

    archivo_local = RAIZ / "accepted_2007_to_2018Q4.csv"
    archivo = archivo_local if archivo_local.exists() else localizar_datos()
    TAMANO_BLOQUE = 50_000
    MAX_CATEGORIAS = 1_000
    inicio = time.perf_counter()
    tipos, columnas = inferir_tipos(archivo)
    numericas = [c for c in columnas if tipos[c] == "float64"]
    texto = [c for c in columnas if tipos[c] == "string"]
    print("Archivo:", archivo.name)
    print("Columnas:", len(columnas))
    print("Versiones:", platform.python_version(), "pandas", pd.__version__, "numpy", np.__version__)

    perfil = {c: {"observados": 0, "faltantes": 0, "minimo": np.inf,
                  "maximo": -np.inf, "todos_enteros": True, "no_finitos": 0}
              for c in numericas}
    vocabularios = {c: set() for c in texto if c not in COLUMNAS_FECHA}
    memoria_original = pd.Series(0, index=columnas, dtype="int64")
    filas = 0
    bloques = 0
    with pd.read_csv(archivo, dtype=tipos, chunksize=TAMANO_BLOQUE,
                     skipinitialspace=True) as lector:
        for bloque in lector:
            filas += len(bloque)
            bloques += 1
            memoria_original += bloque.memory_usage(index=False, deep=True)
            for nombre in numericas:
                serie = bloque[nombre]
                valores = serie.dropna().to_numpy()
                p = perfil[nombre]
                p["observados"] += len(valores)
                p["faltantes"] += int(serie.isna().sum())
                finitos = valores[np.isfinite(valores)]
                p["no_finitos"] += int((~np.isfinite(valores)).sum())
                if len(finitos):
                    p["minimo"] = min(p["minimo"], float(finitos.min()))
                    p["maximo"] = max(p["maximo"], float(finitos.max()))
                    p["todos_enteros"] &= bool((finitos == np.trunc(finitos)).all())
            for nombre, valores in vocabularios.items():
                if valores is not None:
                    valores.update(bloque[nombre].dropna().unique().tolist())
                    if len(valores) > MAX_CATEGORIAS:
                        vocabularios[nombre] = None

    print(f"Archivo completo: {filas:,} filas, {len(columnas)} columnas y {bloques} bloques.")
    print("No se excluyó ningún estado de préstamo.")

    tipos_seguros = dict(tipos)
    for nombre, p in perfil.items():
        if p["observados"] and p["todos_enteros"] and p["no_finitos"] == 0:
            for tipo in ["Int8", "Int16", "Int32"]:
                limites = np.iinfo(tipo.lower())
                if limites.min <= p["minimo"] and p["maximo"] <= limites.max:
                    tipos_seguros[nombre] = tipo
                    break
    for nombre, valores in vocabularios.items():
        if valores:
            tipos_seguros[nombre] = pd.CategoricalDtype(categories=sorted(valores))

    resumen_rangos = pd.DataFrame(perfil).T
    resumen_rangos[["minimo", "maximo"]] = resumen_rangos[["minimo", "maximo"]].replace([np.inf, -np.inf], np.nan)
    resumen_rangos["tipo_original"] = "float64"
    resumen_rangos["tipo_propuesto"] = [str(tipos_seguros[c]) for c in resumen_rangos.index]
    with pd.option_context("display.max_rows", None):
        display(resumen_rangos)

    memoria_segura = pd.Series(0, index=columnas, dtype="int64")
    memoria_float32 = pd.Series(0, index=columnas, dtype="int64")
    verificacion = {c: {"valores_cambiados": 0, "faltantes_cambiados": 0} for c in columnas}
    prueba_float32 = {c: {"observados": 0, "valores_cambiados": 0,
                         "error_absoluto_max": 0.0, "error_relativo_max": 0.0,
                         "nuevos_infinitos": 0, "nuevos_ceros": 0}
                     for c in numericas if tipos_seguros[c] == "float64"}
    filas_verificadas = 0
    with pd.read_csv(archivo, dtype=tipos, chunksize=TAMANO_BLOQUE,
                     skipinitialspace=True) as lector:
        for bloque in lector:
            filas_verificadas += len(bloque)
            for nombre in columnas:
                original = bloque[nombre]
                convertida = original.astype(tipos_seguros[nombre])
                memoria_segura[nombre] += convertida.memory_usage(index=False, deep=True)
                vuelta = convertida.astype(tipos[nombre])
                verificacion[nombre]["faltantes_cambiados"] += int((original.isna() != vuelta.isna()).sum())
                observados = original.notna()
                verificacion[nombre]["valores_cambiados"] += int(
                    (original[observados] != vuelta[observados]).fillna(True).sum())
                experimental = convertida
                if nombre in prueba_float32:
                    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                        experimental = original.astype("float32")
                    a = original[observados].to_numpy()
                    b = experimental[observados].to_numpy(dtype="float64")
                    r = prueba_float32[nombre]
                    r["observados"] += len(a)
                    r["valores_cambiados"] += int((a != b).sum())
                    finitos = np.isfinite(a)
                    r["nuevos_infinitos"] += int((finitos & np.isinf(b)).sum())
                    r["nuevos_ceros"] += int((finitos & (a != 0) & (b == 0)).sum())
                    if finitos.any():
                        errores = np.abs(a[finitos] - b[finitos])
                        r["error_absoluto_max"] = max(r["error_absoluto_max"], float(errores.max()))
                        no_cero = a[finitos] != 0
                        if no_cero.any():
                            relativos = errores[no_cero] / np.abs(a[finitos][no_cero])
                            r["error_relativo_max"] = max(r["error_relativo_max"], float(relativos.max()))
                memoria_float32[nombre] += experimental.memory_usage(index=False, deep=True)

    assert filas_verificadas == filas
    comprobacion = pd.DataFrame(verificacion).T
    assert comprobacion.to_numpy().sum() == 0, "Se detectó una pérdida en la propuesta conservadora."

    for nombre in columnas:
        if memoria_segura[nombre] >= memoria_original[nombre]:
            tipos_seguros[nombre] = tipos[nombre]
            memoria_segura[nombre] = memoria_original[nombre]
            if nombre not in prueba_float32:
                memoria_float32[nombre] = memoria_original[nombre]
    print(f"Verificadas {filas_verificadas:,} filas y {len(columnas)} columnas.")
    print("Cambios de valores en la opción conservadora:", int(comprobacion.valores_cambiados.sum()))
    print("Cambios de faltantes en la opción conservadora:", int(comprobacion.faltantes_cambiados.sum()))

    errores_float32 = pd.DataFrame(prueba_float32).T
    errores_float32["porcentaje_cambiado"] = 100 * errores_float32.valores_cambiados / errores_float32.observados.replace(0, np.nan)

    with pd.option_context("display.max_rows", None, "display.float_format", "{:.8g}".format):
        display(errores_float32.sort_values("error_absoluto_max", ascending=False))
    print("Valores alterados por float32:", int(errores_float32.valores_cambiados.sum()))
    print("Infinitos nuevos:", int(errores_float32.nuevos_infinitos.sum()))
    print("Ceros nuevos:", int(errores_float32.nuevos_ceros.sum()))

    detalle = pd.DataFrame({
        "tipo_original": pd.Series(tipos),
        "tipo_conservador": pd.Series({c: str(tipos_seguros[c]) for c in columnas}),
        "original_MiB": memoria_original / 1024**2,
        "conservador_MiB": memoria_segura / 1024**2,
        "experimental_MiB": memoria_float32 / 1024**2,
    }).join(comprobacion)
    detalle["ahorro_conservador_MiB"] = detalle.original_MiB - detalle.conservador_MiB
    resumen = pd.DataFrame({"memoria_acumulada_MiB": [memoria_original.sum()/1024**2,
        memoria_segura.sum()/1024**2, memoria_float32.sum()/1024**2]},
        index=["Lectura actual", "Conservadora (exacta)", "Experimental (+ float32)"])
    resumen["reduccion_pct"] = 100 * (1 - resumen.memoria_acumulada_MiB / resumen.iloc[0, 0])
    display(resumen.round(2))
    print("Tipos finales propuestos:")
    display(detalle.tipo_conservador.value_counts().to_frame("columnas"))
    with pd.option_context("display.max_rows", None):
        display(detalle.sort_values("ahorro_conservador_MiB", ascending=False).round(4))

    print(f"Filas auditadas: {filas:,}; columnas: {len(columnas)}.")
    print(f"Reducción conservadora acumulada: {resumen.loc['Conservadora (exacta)', 'reduccion_pct']:.2f}%.")
    print("La opción conservadora preservó exactamente valores y faltantes respecto de la lectura actual.")
    if errores_float32.valores_cambiados.sum():
        print("float32 NO conserva exactamente todos los valores: mantener float64 en esas columnas si se exige igualdad exacta.")
    else:
        print("float32 no produjo cambios observables en este archivo bajo esta comparación.")


    import json
    import subprocess
    import tempfile
    from pathlib import Path

    from IPython.display import display
    from src.features import seleccionar_predictoras

    SALIDA_COMPARACION = SALIDAS / "comparacion_carga"
    SALIDA_COMPARACION.mkdir(parents=True, exist_ok=True)

    candidatas = seleccionar_predictoras(pd.DataFrame(columns=columnas), [], False)
    columnas_comparacion = [
        columna for columna in columnas
        if columna in candidatas + ["loan_status", "issue_d"]
    ]
    esquema_comparacion = {}
    for nombre in columnas_comparacion:
        tipo = tipos_seguros[nombre]
        if isinstance(tipo, pd.CategoricalDtype):
            esquema_comparacion[nombre] = {
                "tipo": "category",
                "categorias": tipo.categories.tolist(),
            }
        else:
            esquema_comparacion[nombre] = {"tipo": str(tipo)}

    config_comparacion = {
        "archivo": str(archivo),
        "esquema": esquema_comparacion,
        "filas_fuente_esperadas": filas,
        "bloque": TAMANO_BLOQUE,
    }

    print("Columnas leídas en ambos métodos:", len(columnas_comparacion))
    print("Filas fuente que leerá cada ejecución:", f"{filas:,}")
    print("Tamaño de bloque para el segundo método:", f"{TAMANO_BLOQUE:,} filas")

    def ejecutar_prueba_carga(metodo, repeticion, carpeta_temporal):
        carpeta = carpeta_temporal / f"{metodo}_{repeticion}"
        carpeta.mkdir()
        ruta_config = carpeta / "config.json"
        ruta_config.write_text(json.dumps(config_comparacion), encoding="utf-8")
        ruta_worker = RAIZ / "src" / "benchmark_worker.py"
        assert ruta_worker.exists(), f"No se encontró el programa de medición: {ruta_worker}"

        proceso = subprocess.run(
            [sys.executable, str(ruta_worker), str(ruta_config), metodo, str(carpeta)],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proceso.returncode != 0:
            raise RuntimeError(
                f"Falló la ejecución {metodo} {repeticion}:\n{proceso.stderr[-4000:]}"
            )
        print(proceso.stdout.strip())
        resultado = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
        resultado["repeticion"] = repeticion
        destino = SALIDA_COMPARACION / f"{metodo}_{repeticion}.json"
        destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        return resultado

    orden_pruebas = [
        ("directa", 1), ("bloques", 1),
        ("bloques", 2), ("directa", 2),
        ("directa", 3), ("bloques", 3),
    ]
    resultados_carga = []
    with tempfile.TemporaryDirectory(prefix="comparacion_carga_") as temporal:
        carpeta_temporal = Path(temporal)
        for metodo, repeticion in orden_pruebas:
            print(f"\nMidiendo {metodo}, repetición {repeticion}...")
            resultados_carga.append(
                ejecutar_prueba_carga(metodo, repeticion, carpeta_temporal)
            )
    print("\nMediciones terminadas.")

    tabla_tiempos = pd.DataFrame([
        {
            "método": resultado["metodo"],
            "repetición": resultado["repeticion"],
            "filas_leídas": resultado["filas_fuente"],
            "préstamos_elegibles": resultado["filas_elegibles"],
            "tiempo_segundos": resultado["segundos_carga"],
            "huella": resultado["huella_datos"],
        }
        for resultado in resultados_carga
    ])

    assert tabla_tiempos["filas_leídas"].nunique() == 1
    assert tabla_tiempos["préstamos_elegibles"].nunique() == 1
    assert tabla_tiempos["huella"].nunique() == 1
    print("Las dos formas leyeron y conservaron los mismos datos.")
    display(tabla_tiempos.drop(columns="huella").round(2))

    resumen_tiempos = tabla_tiempos.groupby("método")["tiempo_segundos"].agg(
        repeticiones="count",
        mediana_segundos="median",
        promedio_segundos="mean",
        mínimo_segundos="min",
        máximo_segundos="max",
    ).sort_values("mediana_segundos")
    display(resumen_tiempos.round(2))

    más_rápido = resumen_tiempos.index[0]
    menos_rápido = resumen_tiempos.index[-1]
    mediana_rápido = resumen_tiempos.iloc[0]["mediana_segundos"]
    mediana_lento = resumen_tiempos.iloc[-1]["mediana_segundos"]
    porcentaje = (mediana_lento - mediana_rápido) / mediana_lento * 100
    print(f"Método más rápido por mediana: {más_rápido}.")
    print(f"Su mediana fue {porcentaje:.1f}% menor que la del otro método.")
    return {k: v for k, v in locals().items() if k in ("resumen", "resultados", "resumen_modelos", "resultados_por_corte")}
