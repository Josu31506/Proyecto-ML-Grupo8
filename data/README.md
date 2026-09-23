# Datos

El dataset **no se versiona**: `accepted_2007_to_2018Q4.csv.gz` pesa 392 MB
comprimido (~1.6 GB sin comprimir), muy por encima del limite de GitHub.

## Como obtenerlo

1. Descargar de Kaggle:
   <https://www.kaggle.com/datasets/wordsforthewise/lending-club>
2. Colocar `accepted_2007_to_2018Q4.csv.gz` en `data/raw/`.
3. No hace falta descomprimirlo: pandas lee el `.gz` directamente.

El pipeline tambien lo busca automaticamente en la raiz del proyecto y en
`~/Downloads/`, y acepta el CSV sin comprimir. Ver `src/config.py`.

## Contenido

| Caracteristica | Valor |
|---|---|
| Filas | 2 260 701 |
| Columnas | 151 |
| Periodo | Prestamos otorgados entre 2007 y 2018-Q4 |
| Prestamos con desenlace final | 1 345 310 (`Fully Paid` + `Charged Off`) |

Solo contiene solicitudes **aceptadas**. El archivo de rechazadas
(`rejected_*.csv`) no se usa en este proyecto y tiene un esquema distinto.

## Diccionario de variables

`Lending Club Data Dictionary Approved - ES.csv` describe las 151 columnas en
espanol. Ese archivo si se versiona (16 KB) y esta exceptuado en `.gitignore`.
