# Primera referencia SVM sobre la muestra inicial

Esta carpeta conserva la primera ejecución real autorizada del notebook 05. Procede de la sesión independiente indicada en `referencia_historica.json`; los resultados de otros experimentos permanecen intactos.

La muestra tiene 220,003 préstamos: 73,701 de entrenamiento (2007–2014), 109,382 de validación (2015–2016) y 36,920 de prueba (2017–2018). Los cuatro modelos comparten 65 predictoras y 138 columnas transformadas. El SVM entrega márgenes sin calibrar; ROC-AUC/AP usan `decision_function` y su umbral se eligió mediante F1 en validación.

`metricas_svm_muestra.csv` contiene las doce filas originales sin redondeo. Las configuraciones, variables, particiones, huellas de IDs, tiempos y figura se conservaron desde la ejecución. Los tiempos son informativos y no tienen que repetirse exactamente.

`verificacion.csv` y `manifiesto.json` conservan el estado de la primera ejecución: no había referencia SVM anterior y por eso el contraste se marcó como no verificable. Crear esta referencia no convierte esa primera sesión en una reproducción independiente. Las reproducciones futuras compararán contra estos archivos sin sobrescribirlos.

Los archivos de `outputs/reproducciones/` y el notebook ejecutado permiten auditar la ejecución local; las métricas agregadas de esta carpeta sirven para consulta sin cargar el dataset ni entrenar.
