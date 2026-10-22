"""Entrenamiento con validación temporal y manejo de faltantes sin fuga."""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import evaluate as ev
from . import features as ft
from .config import (ANIO_CORTE_PRUEBA, MAPA_OBJETIVO, RAIZ, SEMILLA,
                     UMBRAL_FALTANTES_FINAL, UMBRAL_INDICADOR, localizar_datos)

# Tipos auditados previamente en el notebook final; se leen las predictoras
# candidatas con enteros compactos para calcular su cobertura sin cargar el CSV
# completo, y después se descartan las que no cumplen el umbral.
TIPOS_AUDITADOS = {'acc_now_delinq': 'Int8', 'acc_open_past_24mths': 'Int8', 'addr_state': 'category', 'all_util': 'Int16', 'annual_inc': 'float64', 'annual_inc_joint': 'float64', 'application_type': 'category', 'avg_cur_bal': 'Int32', 'bc_open_to_buy': 'Int32', 'bc_util': 'float64', 'chargeoff_within_12_mths': 'Int8', 'collection_recovery_fee': 'float64', 'collections_12_mths_ex_med': 'Int8', 'debt_settlement_flag': 'category', 'debt_settlement_flag_date': 'category', 'deferral_term': 'Int8', 'delinq_2yrs': 'Int8', 'delinq_amnt': 'Int32', 'desc': 'string', 'disbursement_method': 'category', 'dti': 'float64', 'dti_joint': 'float64', 'earliest_cr_line': 'string', 'emp_length': 'category', 'emp_title': 'string', 'fico_range_high': 'Int16', 'fico_range_low': 'Int16', 'funded_amnt': 'Int32', 'funded_amnt_inv': 'float64', 'grade': 'category', 'hardship_amount': 'float64', 'hardship_dpd': 'Int8', 'hardship_end_date': 'category', 'hardship_flag': 'category', 'hardship_last_payment_amount': 'float64', 'hardship_length': 'Int8', 'hardship_loan_status': 'category', 'hardship_payoff_balance_amount': 'float64', 'hardship_reason': 'category', 'hardship_start_date': 'category', 'hardship_status': 'category', 'hardship_type': 'category', 'home_ownership': 'category', 'id': 'string', 'il_util': 'Int16', 'initial_list_status': 'category', 'inq_fi': 'Int8', 'inq_last_12m': 'Int8', 'inq_last_6mths': 'Int8', 'installment': 'float64', 'int_rate': 'float64', 'issue_d': 'string', 'last_credit_pull_d': 'category', 'last_fico_range_high': 'Int16', 'last_fico_range_low': 'Int16', 'last_pymnt_amnt': 'float64', 'last_pymnt_d': 'category', 'loan_amnt': 'Int32', 'loan_status': 'category', 'max_bal_bc': 'Int32', 'member_id': 'string', 'mo_sin_old_il_acct': 'Int16', 'mo_sin_old_rev_tl_op': 'Int16', 'mo_sin_rcnt_rev_tl_op': 'Int16', 'mo_sin_rcnt_tl': 'Int16', 'mort_acc': 'Int8', 'mths_since_last_delinq': 'Int16', 'mths_since_last_major_derog': 'Int16', 'mths_since_last_record': 'Int16', 'mths_since_rcnt_il': 'Int16', 'mths_since_recent_bc': 'Int16', 'mths_since_recent_bc_dlq': 'Int16', 'mths_since_recent_inq': 'Int8', 'mths_since_recent_revol_delinq': 'Int16', 'next_pymnt_d': 'category', 'num_accts_ever_120_pd': 'Int8', 'num_actv_bc_tl': 'Int8', 'num_actv_rev_tl': 'Int8', 'num_bc_sats': 'Int8', 'num_bc_tl': 'Int8', 'num_il_tl': 'Int16', 'num_op_rev_tl': 'Int8', 'num_rev_accts': 'Int16', 'num_rev_tl_bal_gt_0': 'Int8', 'num_sats': 'Int8', 'num_tl_120dpd_2m': 'Int8', 'num_tl_30dpd': 'Int8', 'num_tl_90g_dpd_24m': 'Int8', 'num_tl_op_past_12m': 'Int8', 'open_acc': 'Int8', 'open_acc_6m': 'Int8', 'open_act_il': 'Int8', 'open_il_12m': 'Int8', 'open_il_24m': 'Int8', 'open_rv_12m': 'Int8', 'open_rv_24m': 'Int8', 'orig_projected_additional_accrued_interest': 'float64', 'out_prncp': 'float64', 'out_prncp_inv': 'float64', 'payment_plan_start_date': 'category', 'pct_tl_nvr_dlq': 'float64', 'percent_bc_gt_75': 'float64', 'policy_code': 'category', 'pub_rec': 'Int8', 'pub_rec_bankruptcies': 'Int8', 'purpose': 'category', 'pymnt_plan': 'category', 'recoveries': 'float64', 'revol_bal': 'Int32', 'revol_bal_joint': 'Int32', 'revol_util': 'float64', 'sec_app_chargeoff_within_12_mths': 'Int8', 'sec_app_collections_12_mths_ex_med': 'Int8', 'sec_app_earliest_cr_line': 'string', 'sec_app_fico_range_high': 'Int16', 'sec_app_fico_range_low': 'Int16', 'sec_app_inq_last_6mths': 'Int8', 'sec_app_mort_acc': 'Int8', 'sec_app_mths_since_last_major_derog': 'Int16', 'sec_app_num_rev_accts': 'Int8', 'sec_app_open_acc': 'Int8', 'sec_app_open_act_il': 'Int8', 'sec_app_revol_util': 'float64', 'settlement_amount': 'float64', 'settlement_date': 'category', 'settlement_percentage': 'float64', 'settlement_status': 'category', 'settlement_term': 'Int16', 'sub_grade': 'category', 'tax_liens': 'Int8', 'term': 'category', 'title': 'string', 'tot_coll_amt': 'Int32', 'tot_cur_bal': 'Int32', 'tot_hi_cred_lim': 'Int32', 'total_acc': 'Int16', 'total_bal_ex_mort': 'Int32', 'total_bal_il': 'Int32', 'total_bc_limit': 'Int32', 'total_cu_tl': 'Int8', 'total_il_high_credit_limit': 'Int32', 'total_pymnt': 'float64', 'total_pymnt_inv': 'float64', 'total_rec_int': 'float64', 'total_rec_late_fee': 'float64', 'total_rec_prncp': 'float64', 'total_rev_hi_lim': 'Int32', 'url': 'string', 'verification_status': 'category', 'verification_status_joint': 'category', 'zip_code': 'category'}

NOMBRES_VARIABLES = {'acc_now_delinq': 'cuentas_en_mora_actual', 'acc_open_past_24mths': 'cuentas_abiertas_ultimos_24m', 'annual_inc': 'ingreso_anual', 'avg_cur_bal': 'saldo_promedio_cuentas', 'bc_open_to_buy': 'credito_disponible_tarjetas', 'bc_util': 'utilizacion_tarjetas_pct', 'chargeoff_within_12_mths': 'cuentas_castigadas_12m', 'collections_12_mths_ex_med': 'cobranzas_no_medicas_12m', 'delinq_2yrs': 'moras_ultimos_2_anios', 'delinq_amnt': 'monto_en_mora_actual', 'dti': 'ratio_deuda_ingreso', 'fico_range_high': 'fico_limite_superior', 'fico_range_low': 'fico_limite_inferior', 'inq_last_6mths': 'consultas_credito_6m', 'loan_amnt': 'monto_prestamo', 'mo_sin_old_il_acct': 'meses_desde_cuenta_plazos_mas_antigua', 'mo_sin_old_rev_tl_op': 'meses_desde_revolvente_mas_antigua', 'mo_sin_rcnt_rev_tl_op': 'meses_desde_revolvente_mas_reciente', 'mo_sin_rcnt_tl': 'meses_desde_cuenta_mas_reciente', 'mort_acc': 'cuentas_hipotecarias', 'mths_since_last_delinq': 'meses_desde_ultima_mora', 'mths_since_last_major_derog': 'meses_desde_ultima_mora_grave', 'mths_since_last_record': 'meses_desde_ultimo_registro_publico', 'mths_since_recent_bc': 'meses_desde_tarjeta_mas_reciente', 'mths_since_recent_bc_dlq': 'meses_desde_ultima_mora_tarjeta', 'mths_since_recent_inq': 'meses_desde_ultima_consulta_credito', 'mths_since_recent_revol_delinq': 'meses_desde_ultima_mora_revolvente', 'num_accts_ever_120_pd': 'cuentas_con_mora_120_dias', 'num_actv_bc_tl': 'tarjetas_bancarias_activas', 'num_actv_rev_tl': 'cuentas_revolventes_activas', 'num_bc_sats': 'tarjetas_bancarias_al_dia', 'num_bc_tl': 'total_cuentas_tarjeta', 'num_il_tl': 'total_cuentas_plazos', 'num_op_rev_tl': 'cuentas_revolventes_abiertas', 'num_rev_accts': 'total_cuentas_revolventes', 'num_rev_tl_bal_gt_0': 'revolventes_con_saldo', 'num_sats': 'cuentas_satisfactorias', 'num_tl_120dpd_2m': 'cuentas_mora_120_dias_ultimos_2m', 'num_tl_30dpd': 'cuentas_mora_30_dias_actual', 'num_tl_90g_dpd_24m': 'cuentas_mora_90_mas_dias_24m', 'num_tl_op_past_12m': 'cuentas_abiertas_12m', 'open_acc': 'lineas_credito_abiertas', 'pct_tl_nvr_dlq': 'pct_cuentas_sin_mora', 'percent_bc_gt_75': 'pct_tarjetas_utilizacion_mayor_75', 'pub_rec': 'registros_publicos_adversos', 'pub_rec_bankruptcies': 'quiebras_registradas', 'revol_bal': 'saldo_credito_revolvente', 'revol_util': 'utilizacion_credito_revolvente_pct', 'tax_liens': 'gravamenes_fiscales', 'term': 'plazo_meses', 'tot_coll_amt': 'monto_total_cobranzas', 'tot_cur_bal': 'saldo_total_actual', 'tot_hi_cred_lim': 'limite_credito_total', 'total_acc': 'total_lineas_credito', 'total_bal_ex_mort': 'saldo_total_sin_hipoteca', 'total_bc_limit': 'limite_total_tarjetas', 'total_il_high_credit_limit': 'limite_total_creditos_plazos', 'total_rev_hi_lim': 'limite_total_credito_revolvente', 'emp_length': 'antiguedad_laboral', 'home_ownership': 'tenencia_vivienda', 'verification_status': 'verificacion_ingresos', 'purpose': 'proposito_prestamo', 'addr_state': 'estado_residencia', 'application_type': 'tipo_solicitud', 'earliest_cr_line': 'fecha_primera_linea_credito', 'issue_d': 'fecha_otorgamiento', 'loan_status': 'estado_prestamo', 'verification_status_joint': 'verificacion_ingresos_conjunta', 'all_util': 'utilizacion_global_pct', 'annual_inc_joint': 'ingreso_anual_conjunto', 'dti_joint': 'ratio_deuda_ingreso_conjunto', 'open_acc_6m': 'creditos_abiertos_6m', 'open_act_il': 'creditos_plazo_activos', 'open_il_12m': 'creditos_plazo_abiertos_12m', 'open_il_24m': 'creditos_plazo_abiertos_24m', 'mths_since_rcnt_il': 'meses_desde_credito_plazo_reciente', 'total_bal_il': 'saldo_total_creditos_plazo', 'il_util': 'utilizacion_creditos_plazo_pct', 'open_rv_12m': 'creditos_revolventes_abiertos_12m', 'open_rv_24m': 'creditos_revolventes_abiertos_24m', 'max_bal_bc': 'saldo_maximo_tarjetas', 'inq_fi': 'consultas_financieras', 'total_cu_tl': 'cuentas_credito_uniones', 'inq_last_12m': 'consultas_credito_12m', 'revol_bal_joint': 'saldo_revolvente_conjunto', 'sec_app_fico_range_low': 'fico_inferior_solicitante_secundario', 'sec_app_fico_range_high': 'fico_superior_solicitante_secundario', 'sec_app_earliest_cr_line': 'fecha_credito_mas_antiguo_secundario', 'sec_app_inq_last_6mths': 'consultas_credito_6m_secundario', 'sec_app_mort_acc': 'cuentas_hipotecarias_secundario', 'sec_app_open_acc': 'lineas_credito_abiertas_secundario', 'sec_app_revol_util': 'utilizacion_revolvente_secundario_pct', 'sec_app_open_act_il': 'creditos_plazo_activos_secundario', 'sec_app_num_rev_accts': 'cuentas_revolventes_secundario', 'sec_app_chargeoff_within_12_mths': 'castigos_12m_secundario', 'sec_app_collections_12_mths_ex_med': 'cobranzas_12m_secundario', 'sec_app_mths_since_last_major_derog': 'meses_desde_mora_grave_secundario', 'meses_desde_earliest_cr_line': 'antiguedad_historial_crediticio_meses', 'meses_desde_sec_app_earliest_cr_line': 'antiguedad_historial_secundario_meses', 'default': 'incumplimiento'}


def cargar_datos_modelado(ruta=None):
    """Carga las predictoras con <10% de faltantes hasta el primer corte.

    La lista fija se determina con préstamos elegibles hasta 2012, que forman
    el primer entrenamiento temporal. Así se puede usar el mismo conjunto de
    variables en los cuatro cortes sin consultar años de validación o prueba.
    """
    ruta = localizar_datos(ruta) if ruta is not None else localizar_datos()
    columnas_fuente = pd.read_csv(ruta, nrows=0).columns.tolist()
    candidatas = ft.seleccionar_predictoras(
        pd.DataFrame(columns=columnas_fuente), [], incluir_precio_lc=False)
    columnas = candidatas + ["loan_status", "issue_d"]

    dtype_lectura = {}
    for columna in columnas:
        sugerido = TIPOS_AUDITADOS.get(columna)
        if sugerido is None:
            raise ValueError(f"No existe tipo auditado para {columna}.")
        dtype_lectura[columna] = "string" if sugerido in ["category", "string"] else sugerido

    datos = pd.read_csv(ruta, usecols=columnas, dtype=dtype_lectura,
                        skipinitialspace=True, low_memory=False)
    for columna in datos.select_dtypes(include="string").columns:
        datos[columna] = datos[columna].astype("category")
    for columna in ["issue_d", "earliest_cr_line", "sec_app_earliest_cr_line"]:
        if columna in datos:
            original_no_vacio = datos[columna].notna()
            datos[columna] = pd.to_datetime(datos[columna], format="%b-%Y", errors="coerce")
            if (original_no_vacio & datos[columna].isna()).any():
                raise ValueError(f"Hay valores de fecha no reconocidos en {columna}.")
    if "term" in datos:
        datos["term"] = pd.to_numeric(
            datos["term"].astype("string").str.strip().str.replace(" months", "", regex=False),
            errors="raise").astype("Int8")

    elegibles = datos["loan_status"].isin(MAPA_OBJETIVO) & datos["issue_d"].notna()
    datos = datos.loc[elegibles].copy()
    datos["anio"] = datos["issue_d"].dt.year.astype("int16")
    if datos["anio"].isna().any():
        raise ValueError("Hay préstamos elegibles sin año de otorgamiento.")

    mascara_primer_entrenamiento = datos["anio"] <= 2012
    predictoras, excluidas, tabla_faltantes = ft.seleccionar_por_cobertura(
        datos.loc[mascara_primer_entrenamiento], candidatas, UMBRAL_FALTANTES_FINAL)
    if not predictoras:
        raise ValueError("El umbral de faltantes excluyó todas las predictoras.")
    print(
        f"Selección por faltantes hasta 2012 (<{UMBRAL_FALTANTES_FINAL}%): "
        f"{len(predictoras)} conservadas de {len(candidatas)}; "
        f"{len(excluidas)} excluidas."
    )

    datos["default"] = datos["loan_status"].map(MAPA_OBJETIVO).astype("int8")
    datos["fecha_prestamo"] = datos["issue_d"]
    columnas_mantener = predictoras + ["default", "fecha_prestamo", "anio"]
    datos = datos[columnas_mantener]
    return datos, predictoras


def crear_cortes_temporales(datos, primer_anio_validacion=2013,
                             ultimo_anio_validacion=2016,
                             anio_prueba=ANIO_CORTE_PRUEBA):
    """Crea validaciones expansivas: años previos entrenan, el siguiente valida."""
    cortes = []
    for anio_validacion in range(primer_anio_validacion, ultimo_anio_validacion + 1):
        anio_final_train = anio_validacion - 1
        n_train = int((datos["anio"] <= anio_final_train).sum())
        mascara_valid = datos["anio"] == anio_validacion
        n_valid = int(mascara_valid.sum())
        clases = datos.loc[mascara_valid, "default"].value_counts()
        if n_train == 0 or n_valid == 0 or clases.size < 2:
            raise ValueError(f"Corte sin datos o sin ambas clases: validación {anio_validacion}.")
        cortes.append({
            "fold": f"Validación {anio_validacion}",
            "anio_final_entrenamiento": anio_final_train,
            "anio_validacion": anio_validacion,
            "n_entrenamiento": n_train,
            "n_validacion": n_valid,
            "default_validacion_pct": float(datos.loc[mascara_valid, "default"].mean() * 100),
        })
    mascara_prueba = datos["anio"] >= anio_prueba
    cortes.append({
        "fold": "Prueba reservada",
        "anio_final_entrenamiento": np.nan,
        "anio_validacion": f">={anio_prueba}",
        "n_entrenamiento": int((datos["anio"] < anio_prueba).sum()),
        "n_validacion": int(mascara_prueba.sum()),
        # En la fila de prueba mostramos solo su tamaño; no inspeccionamos sus etiquetas.
        "default_validacion_pct": np.nan,
    })
    return pd.DataFrame(cortes)


def crear_tablas_imputadas_corte(datos, predictoras, anio_final_entrenamiento,
                                  anio_validacion, usar_indicadores=True,
                                  umbral_indicador=UMBRAL_INDICADOR):
    """Crea tablas imputadas para un corte; las medianas salen solo de train.

    Devuelve X de train/validación, y de ambos conjuntos, medianas usadas,
    columnas con indicador y tablas separadas de indicadores 0/1. Las tablas
    grandes se deben procesar y liberar corte por corte.
    """
    mascara_train = datos["anio"] <= anio_final_entrenamiento
    mascara_valid = datos["anio"] == anio_validacion
    if (datos.loc[mascara_train, "anio"].max() >= anio_validacion
            or datos.loc[mascara_valid].empty):
        raise ValueError("Los años de entrenamiento y validación se superponen o están vacíos.")
    columnas_entrada = predictoras + ["fecha_prestamo"]
    train = ft.preparar_matriz(datos.loc[mascara_train, columnas_entrada]).drop(columns="fecha_prestamo")
    valid = ft.preparar_matriz(datos.loc[mascara_valid, columnas_entrada]).drop(columns="fecha_prestamo")
    y_train = datos.loc[mascara_train, "default"].astype("int8").copy()
    y_valid = datos.loc[mascara_valid, "default"].astype("int8").copy()
    if not train.index.equals(y_train.index) or not valid.index.equals(y_valid.index):
        raise AssertionError("Los predictores y el objetivo perdieron alineación.")

    numericas = train.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in train.columns if c not in numericas]
    # Tratamos infinitos como faltantes antes de calcular las medianas.
    train = train.copy()
    valid = valid.copy()
    for columna in numericas:
        train[columna] = pd.to_numeric(train[columna], errors="coerce").astype("float64")
        valid[columna] = pd.to_numeric(valid[columna], errors="coerce").astype("float64")
        train[columna] = train[columna].replace([np.inf, -np.inf], np.nan)
        valid[columna] = valid[columna].replace([np.inf, -np.inf], np.nan)

    porcentaje_faltante_train = train.isna().mean() * 100
    columnas_indicadoras = [
        c for c in numericas
        if usar_indicadores and porcentaje_faltante_train[c] > umbral_indicador
    ]
    indicadores_train = train[columnas_indicadoras].isna().astype("int8")
    indicadores_valid = valid[columnas_indicadoras].isna().astype("int8")

    medianas = train[numericas].median()
    if medianas.isna().any():
        sin_mediana = medianas[medianas.isna()].index.tolist()
        raise ValueError(f"Hay columnas numéricas sin mediana en el entrenamiento: {sin_mediana}")
    for columna in numericas:
        train[columna] = train[columna].fillna(medianas[columna])
        valid[columna] = valid[columna].fillna(medianas[columna])
    for columna in categoricas:
        train[columna] = train[columna].astype(object).where(train[columna].notna(), "Sin información")
        valid[columna] = valid[columna].astype(object).where(valid[columna].notna(), "Sin información")

    indicadores_train.columns = [f"{c}__faltante" for c in indicadores_train.columns]
    indicadores_valid.columns = indicadores_train.columns
    train = pd.concat([train, indicadores_train], axis=1)
    valid = pd.concat([valid, indicadores_valid], axis=1)
    resumen_faltantes = pd.DataFrame({
        "porcentaje_faltante_entrenamiento": porcentaje_faltante_train,
        "tratamiento": [
            "mediana + indicador" if c in columnas_indicadoras else
            "mediana" if c in numericas else "Sin información"
            for c in train.columns if not c.endswith("__faltante")
        ],
    }, index=[c for c in train.columns if not c.endswith("__faltante")])
    return {
        "X_entrenamiento": train,
        "X_validacion": valid,
        "y_entrenamiento": y_train,
        "y_validacion": y_valid,
        "medianas": medianas,
        "columnas_indicadoras": columnas_indicadoras,
        "indicadores_entrenamiento": indicadores_train,
        "indicadores_validacion": indicadores_valid,
        "resumen_faltantes": resumen_faltantes,
    }


def construir_codificador(X_entrenamiento):
    """Ajusta escalado y one-hot solo con entrenamiento; conserva matriz dispersa."""
    numericas = X_entrenamiento.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in X_entrenamiento.columns if c not in numericas]
    transformadores = []
    if numericas:
        transformadores.append(("numericas", StandardScaler(with_mean=False), numericas))
    if categoricas:
        transformadores.append((
            "categoricas",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True),
            categoricas,
        ))
    return ColumnTransformer(transformadores, remainder="drop", sparse_threshold=1.0)


def modelos_iniciales(semilla=SEMILLA):
    """Configuraciones de partida; se pueden sustituir o ampliar desde el notebook."""
    return {
        "baseline_mayoria": DummyClassifier(strategy="most_frequent"),
        "regresion_logistica": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=semilla),
        "svm_lineal": LinearSVC(
            C=1.0, dual="auto", max_iter=5000, random_state=semilla),
        "arbol_decision": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=0.005, random_state=semilla),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=75, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, early_stopping=False, random_state=semilla),
    }


def _evaluar_puntuaciones(y_true, puntuacion, estimador, umbral_probabilidad):
    """Evalúa probabilidades o márgenes de decisión de SVM lineal."""
    if hasattr(estimador, "predict_proba"):
        return ev.evaluar(y_true, puntuacion, umbral_probabilidad)
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score,
                                 roc_auc_score, average_precision_score)
    umbral = 0.0
    prediccion = (puntuacion >= umbral).astype("int8")
    matriz = confusion_matrix(y_true, prediccion, labels=[0, 1])
    vn, fp, fn, vp = matriz.ravel()
    return {
        "n": int(len(y_true)),
        "tasa_default_real": float(np.mean(y_true)),
        "umbral": umbral,
        "roc_auc": float(roc_auc_score(y_true, puntuacion)),
        "pr_auc": float(average_precision_score(y_true, puntuacion)),
        "ks": ev.estadistico_ks(y_true, puntuacion),
        "brier": np.nan,
        "accuracy": float(accuracy_score(y_true, prediccion)),
        "precision": float(precision_score(y_true, prediccion, zero_division=0)),
        "recall": float(recall_score(y_true, prediccion, zero_division=0)),
        "f1": float(f1_score(y_true, prediccion, zero_division=0)),
        "matriz_confusion": {"vn": int(vn), "fp": int(fp),
                             "fn": int(fn), "vp": int(vp)},
    }


def evaluar_modelos_temporales(datos, predictoras, modelos=None,
                                primer_anio_validacion=2013,
                                ultimo_anio_validacion=2016,
                                usar_indicadores=True,
                                umbral_indicador=UMBRAL_INDICADOR,
                                umbral_clasificacion=0.5, verbose=True):
    """Compara modelos en validación expansiva y nunca entrena con validación.

    La imputación se aprende por corte. El 2017+ no se usa aquí: queda reservado
    para una evaluación final posterior a elegir el modelo.
    """
    modelos = modelos_iniciales() if modelos is None else modelos
    resultados = []
    resumen_medianas = []
    for anio_validacion in range(primer_anio_validacion, ultimo_anio_validacion + 1):
        anio_final_train = anio_validacion - 1
        tablas = crear_tablas_imputadas_corte(
            datos, predictoras, anio_final_train, anio_validacion,
            usar_indicadores=usar_indicadores,
            umbral_indicador=umbral_indicador)
        codificador = construir_codificador(tablas["X_entrenamiento"])
        X_train = codificador.fit_transform(tablas["X_entrenamiento"])
        X_valid = codificador.transform(tablas["X_validacion"])
        y_train = tablas["y_entrenamiento"].to_numpy()
        y_valid = tablas["y_validacion"].to_numpy()
        if len(np.unique(y_valid)) < 2:
            raise ValueError(f"Validación {anio_validacion} no contiene ambas clases.")

        for nombre, estimador_base in modelos.items():
            inicio = time.perf_counter()
            estimador = clone(estimador_base)
            if isinstance(estimador, HistGradientBoostingClassifier):
                X_train_model = X_train.toarray() if sparse.issparse(X_train) else X_train
                X_valid_model = X_valid.toarray() if sparse.issparse(X_valid) else X_valid
            else:
                X_train_model, X_valid_model = X_train, X_valid
            estimador.fit(X_train_model, y_train)
            segundos = time.perf_counter() - inicio
            if hasattr(estimador, "predict_proba"):
                puntuacion = estimador.predict_proba(X_valid_model)[:, 1]
            elif hasattr(estimador, "decision_function"):
                puntuacion = estimador.decision_function(X_valid_model)
            else:
                raise TypeError(f"{nombre} debe exponer probabilidades o una puntuación de decisión.")
            metricas = _evaluar_puntuaciones(
                y_valid, puntuacion, estimador, umbral_clasificacion)
            resultados.append({
                "modelo": nombre,
                "indicadores_faltantes": usar_indicadores,
                "corte": f"train≤{anio_final_train} / validación {anio_validacion}",
                "anio_final_entrenamiento": anio_final_train,
                "anio_validacion": anio_validacion,
                "segundos_entrenamiento": segundos,
                **metricas,
            })
            if isinstance(estimador, HistGradientBoostingClassifier):
                del X_train_model, X_valid_model
            del estimador, puntuacion
            gc.collect()
        resumen_medianas.append(
            tablas["medianas"].rename(f"entrenamiento_hasta_{anio_final_train}"))
        if verbose:
            print(f"Corte completado: entrenamiento hasta {anio_final_train}; validación {anio_validacion}.", flush=True)
        del tablas, codificador, X_train, X_valid, y_train, y_valid
        gc.collect()
    return pd.DataFrame(resultados), pd.concat(resumen_medianas, axis=1)


def resumir_metricas(resultados):
    """Promedio y desviación por modelo y uso de indicadores de faltantes."""
    columnas = ["roc_auc", "pr_auc", "ks", "recall", "precision", "f1", "accuracy",
                "segundos_entrenamiento"]
    grupos = ["modelo"]
    if "indicadores_faltantes" in resultados.columns:
        grupos.append("indicadores_faltantes")
    resumen = resultados.groupby(grupos)[columnas].agg(["mean", "std"])
    resumen.columns = [f"{metrica}_{estadistico}" for metrica, estadistico in resumen.columns]
    return resumen.sort_values("roc_auc_mean", ascending=False)
