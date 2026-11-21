# Datos de Lending Club

Esta carpeta contiene las instrucciones para obtener el dataset y el diccionario de variables utilizado en el proyecto.

## Cómo agregar el dataset

1. Descargar el conjunto de datos desde [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
2. Si la descarga viene en un archivo `.zip`, abrirlo y extraer **`accepted_2007_to_2018Q4.csv.gz`** (o su versión `.csv`).
3. Crear la carpeta `data/raw/` si todavía no existe.
4. Colocar el archivo extraído dentro de `data/raw/`.

La estructura local recomendada es:

```text
data/
├── README.md
├── Lending Club Data Dictionary Approved - ES.csv
└── raw/
    └── accepted_2007_to_2018Q4.csv.gz
```

**No es necesario descomprimir el archivo `.csv.gz`**: el proyecto puede leerlo directamente. También se admite `accepted_2007_to_2018Q4.csv` sin comprimir.

El archivo `.zip` descargado de Kaggle **no se procesa directamente**: primero hay que extraer de él el CSV o CSV.GZ.

**Importante:** `data/raw/` está excluida de Git mediante `.gitignore`. El dataset debe colocarse **localmente**; no debe subirse al repositorio de GitHub.

## Diccionario de variables

El archivo [**Lending Club Data Dictionary Approved - ES.csv**](<Lending Club Data Dictionary Approved - ES.csv>) contiene los nombres originales de las variables de Lending Club y sus descripciones en español.

Sirve para consultar el significado de las columnas del dataset. **Este diccionario sí está incluido en el repositorio**, por lo que puede abrirse sin descargar los datos originales.
