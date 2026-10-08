# BankMarketing — Análisis Exploratorio de Datos (EDA)

## Descripción
Aplicación interactiva desarrollada con Streamlit para explorar el dataset `BankMarketing.csv`. Incluye carga y validación del CSV, clasificación de variables, estadísticas descriptivas, revisión de valores faltantes, gráficos univariados y bivariados, análisis dinámico y conclusiones descriptivas. No construye modelos predictivos.

## Tecnologías
- Python
- Streamlit
- Pandas y NumPy
- Matplotlib y Seaborn

## Archivos del proyecto
- `app.py`: aplicación principal.
- `requirements.txt`: dependencias.
- `BankMarketing.csv`: dataset del caso de estudio.

## Ejecución local
1. Instala Python 3.10 o superior.
2. Descarga/clona el repositorio y coloca `BankMarketing.csv` en la carpeta del proyecto.
3. Abre una terminal en esa carpeta.
4. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

5. Ejecuta la aplicación:

   ```bash
   streamlit run app.py
   ```

6. En el menú lateral, entra en **Carga del dataset** y sube `BankMarketing.csv`.

## Publicación en Streamlit Community Cloud
1. Sube `app.py`, `requirements.txt`, `README.md` y `BankMarketing.csv` a un repositorio de GitHub.
2. Ingresa a https://share.streamlit.io/ y conecta tu cuenta de GitHub.
3. Selecciona el repositorio y el archivo principal `app.py`.
4. Despliega la aplicación y verifica que el enlace público funcione.

## Enlaces
- Repositorio GitHub: [Agregar enlace después de publicar]
- Aplicación Streamlit: [Agregar enlace después de desplegar]

## Nota metodológica
Los resultados son descriptivos. Las relaciones observadas no demuestran causalidad y deben interpretarse considerando el contexto comercial, la calidad de los datos y el tamaño de las muestras.
