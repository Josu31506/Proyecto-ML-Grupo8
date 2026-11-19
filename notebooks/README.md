# Notebooks oficiales y suplementarios

El índice y las instrucciones vigentes están en [README principal](../README.md). Los notebooks 01–10 son la secuencia oficial; `11_svm_muestra.ipynb` es un suplemento para dejar explícita la comparación del SVM sobre la muestra inicial de 220,003 préstamos.

Para regenerar experimentos temporales en 05–07, activar únicamente el control correspondiente. Las implementaciones reutilizables están en los módulos `comparacion_*.py`, `sensibilidad_faltantes.py`, `entrenamiento_inicial.py` y `auditoria_memoria.py` de `../src/`; las nuevas ejecuciones escriben en `../outputs/reproducciones/`. Los CSV históricos no se sobrescriben con ese mecanismo.

Los reportes 06 y 07 fueron ejecutados en modo de lectura de evidencia guardada. Esto verifica sus tablas y gráficos, no representa un nuevo entrenamiento de los modelos.
