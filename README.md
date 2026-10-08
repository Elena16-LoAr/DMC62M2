import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="BankMarketing EDA",
    page_icon="📊",
    layout="wide"
)

st.title("Análisis Exploratorio de Datos - BankMarketing")

# 1. Carga del dataset
archivo = st.file_uploader(
    "Carga el archivo BankMarketing.csv",
    type=["csv"]
)

if archivo is not None:
    df = pd.read_csv(archivo, sep=";")

    st.success("Dataset cargado correctamente")
    st.write("Dimensiones:", df.shape)
    st.dataframe(df.head())

    # 2. Controles en el menú lateral
    st.sidebar.header("Filtros del análisis")

    columna = st.sidebar.selectbox(
        "Selecciona una variable",
        options=df.columns.tolist()
    )

    columnas = st.sidebar.multiselect(
        "Selecciona variables para comparar",
        options=df.columns.tolist(),
        default=["age", "duration", "y"]
    )

    edad_min = int(df["age"].min())
    edad_max = int(df["age"].max())

    rango_edad = st.sidebar.slider(
        "Rango de edad",
        min_value=edad_min,
        max_value=edad_max,
        value=(edad_min, edad_max)
    )

    solo_aceptaron = st.sidebar.checkbox(
        "Mostrar solo clientes que aceptaron",
        value=False
    )

    # 3. Aplicación de filtros
    datos = df[
        df["age"].between(rango_edad[0], rango_edad[1])
    ].copy()

    if solo_aceptaron:
        datos = datos[datos["y"] == "yes"]

    # 4. Pestañas del análisis
    tab1, tab2, tab3 = st.tabs([
        "Datos filtrados",
        "Estadísticas",
        "Variables seleccionadas"
    ])

    with tab1:
        st.subheader("Vista de los datos")
        st.write(f"Registros encontrados: {len(datos)}")
        st.dataframe(datos)

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Clientes", len(datos))

        with col2:
            st.metric(
                "Edad promedio",
                round(datos["age"].mean(), 2)
                if not datos.empty else "Sin datos"
            )

        with col3:
            tasa = (
                (datos["y"] == "yes").mean() * 100
                if not datos.empty else 0
            )
            st.metric("Aceptación (%)", f"{tasa:.2f}%")

    with tab3:
        if columnas:
            st.dataframe(datos[columnas])
        else:
            st.info("Selecciona al menos una variable.")

else:
    st.info("Carga el archivo CSV para comenzar el análisis.")
