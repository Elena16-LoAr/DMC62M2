import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(
    page_title="BankMarketing | EDA",
    page_icon="📊",
    layout="wide"
)

# ----------------------------- CONFIGURACIÓN -----------------------------
st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e4e9f2;
        padding: 14px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("BankMarketing | Análisis exploratorio de datos")
st.caption("Exploración de la campaña de marketing de una institución financiera")

# ----------------------------- CLASE POO ---------------------------------
class DataAnalyzer:
    """Encapsula tareas frecuentes de análisis exploratorio."""

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(exclude=np.number).columns.tolist()
        return numericas, categoricas

    def resumen_nulos(self):
        resultado = self.df.isna().sum().to_frame("valores_nulos")
        resultado["porcentaje"] = (resultado["valores_nulos"] / max(len(self.df), 1) * 100).round(2)
        return resultado.sort_values("valores_nulos", ascending=False)

    def estadisticas(self):
        return self.df.describe(include="all").T

    def tasa_aceptacion(self):
        if "y" not in self.df.columns or self.df.empty:
            return 0.0
        y = self.df["y"].astype(str).str.strip().str.lower()
        return float((y == "yes").mean() * 100)

    def moda(self, columna):
        valores = self.df[columna].mode(dropna=True)
        return valores.iloc[0] if not valores.empty else "Sin moda"


def leer_csv(archivo):
    """Lee CSV con separador detectado automáticamente y normaliza encabezados."""
    contenido = archivo.getvalue()
    try:
        df = pd.read_csv(StringIO(contenido.decode("utf-8-sig")), sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(StringIO(contenido.decode("latin-1")), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def grafico_barras_conteo(data, columna, top_n=20):
    conteo = data[columna].astype("string").fillna("Nulo").value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=conteo.values, y=conteo.index.astype(str), ax=ax, color="#3478c8")
    ax.set_title(f"Frecuencia de {columna}")
    ax.set_xlabel("Número de registros")
    ax.set_ylabel(columna)
    fig.tight_layout()
    return fig


def grafico_histograma(data, columna):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data[columna].dropna(), bins=30, kde=True, ax=ax, color="#3478c8")
    ax.set_title(f"Distribución de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def interpretar_distribucion(serie, nombre):
    serie = pd.to_numeric(serie, errors="coerce").dropna()
    if serie.empty:
        return f"No hay valores numéricos suficientes para interpretar {nombre}."
    media, mediana = serie.mean(), serie.median()
    if media > mediana * 1.1:
        forma = "presenta indicios de asimetría hacia valores altos"
    elif media < mediana * 0.9:
        forma = "presenta indicios de asimetría hacia valores bajos"
    else:
        forma = "tiene una media relativamente cercana a la mediana"
    return (f"En {nombre}, la media es {media:,.2f} y la mediana {mediana:,.2f}; "
            f"la distribución {forma}. Interpreta este resultado junto con el histograma.")


def validar_columnas(df):
    requeridas = {"age", "y"}
    faltantes = requeridas - set(df.columns)
    return faltantes


# ----------------------------- NAVEGACIÓN --------------------------------
pagina = st.sidebar.radio(
    "Menú principal",
    ["Home", "Carga del dataset", "Análisis exploratorio (EDA)", "Conclusiones"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto académico · Python for Analytics")

if pagina == "Home":
    st.header("Presentación del proyecto")
    st.write(
        "Esta aplicación explora el dataset BankMarketing para describir a los clientes, "
        "examinar la aceptación de la campaña y encontrar patrones útiles para la toma "
        "de decisiones comerciales. El objetivo es realizar análisis exploratorio; no se "
        "construyen modelos predictivos."
    )
    a, b = st.columns(2)
    with a:
        st.subheader("Datos del proyecto")
        st.write("**Autor/a:** [Escribe tu nombre completo]")
        st.write("**Curso:** Especialización en Python for Analytics")
        st.write("**Año:** 2026")
    with b:
        st.subheader("Dataset")
        st.write(
            "Incluye variables demográficas, laborales, de contacto, historial de campañas "
            "e indicadores económicos. La variable `y` indica si el cliente aceptó la oferta "
            "(`yes`) o no la aceptó (`no`)."
        )
    st.subheader("Tecnologías utilizadas")
    st.write("Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn")
    st.info("Ve a «Carga del dataset» y sube BankMarketing.csv para habilitar los análisis.")

elif pagina == "Carga del dataset":
    st.header("Carga y validación del dataset")
    archivo = st.file_uploader("Selecciona BankMarketing.csv", type=["csv"])
    if archivo is None:
        st.warning("Primero carga el archivo CSV. Los análisis se habilitan después de la carga.")
        st.stop()
    try:
        df = leer_csv(archivo)
        if df.empty:
            st.error("El archivo se leyó, pero no contiene registros.")
            st.stop()
        st.session_state["bankmarketing_df"] = df
        st.success("Archivo cargado correctamente.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", f"{df.shape[1]:,}")
        c3.metric("Tamaño", f"{archivo.size / (1024 * 1024):.2f} MB")
        st.subheader("Vista previa")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Tipos de datos detectados")
        tipos = df.dtypes.astype(str).rename("tipo_de_dato").to_frame()
        st.dataframe(tipos, use_container_width=True)
        faltantes = validar_columnas(df)
        if faltantes:
            st.warning("No se encontraron algunas columnas habituales: " + ", ".join(sorted(faltantes)) +
                       ". La aplicación seguirá funcionando en los análisis compatibles.")
    except Exception as e:
        st.error(f"No se pudo leer el archivo. Verifica que sea un CSV válido. Detalle: {e}")

elif pagina == "Análisis exploratorio (EDA)":
    if "bankmarketing_df" not in st.session_state:
        st.warning("Primero carga el dataset desde «Carga del dataset».")
        st.stop()

    df_original = st.session_state["bankmarketing_df"].copy()
    analyzer_original = DataAnalyzer(df_original)
    numericas_original, categoricas_original = analyzer_original.clasificar_variables()

    st.sidebar.subheader("Filtros globales")
    datos = df_original.copy()

    if "age" in datos.columns and pd.api.types.is_numeric_dtype(datos["age"]):
        minimo, maximo = int(datos["age"].min()), int(datos["age"].max())
        rango = st.sidebar.slider("Rango de edad", minimo, maximo, (minimo, maximo))
        datos = datos[datos["age"].between(rango[0], rango[1])]

    if "y" in datos.columns:
        solo_yes = st.sidebar.checkbox("Mostrar solo clientes que aceptaron (y = yes)", value=False)
        if solo_yes:
            datos = datos[datos["y"].astype(str).str.lower().str.strip() == "yes"]

    st.sidebar.caption(f"Registros después de filtros: {len(datos):,} de {len(df_original):,}")
    if datos.empty:
        st.warning("Los filtros seleccionados no dejan registros. Amplía el rango o desactiva el filtro.")
        st.stop()

    analyzer = DataAnalyzer(datos)
    numericas, categoricas = analyzer.clasificar_variables()

    tabs = st.tabs([
        "1. Información general", "2. Tipos de variables", "3. Estadísticas",
        "4. Valores faltantes", "5. Distribución numérica", "6. Categóricas",
        "7. Numérica vs resultado", "8. Categórica vs resultado",
        "9. Análisis dinámico", "10. Hallazgos clave"
    ])

    with tabs[0]:
        st.subheader("Ítem 1. Información general del dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros analizados", f"{len(datos):,}")
        c2.metric("Variables", len(datos.columns))
        c3.metric("Tasa de aceptación", f"{analyzer.tasa_aceptacion():.2f}%" if "y" in datos else "No disponible")
        st.markdown("**Estructura e información equivalente a `DataFrame.info()`**")
        buffer = StringIO()
        datos.info(buf=buffer)
        st.code(buffer.getvalue(), language="text")
        st.dataframe(datos.dtypes.astype(str).rename("tipo").to_frame(), use_container_width=True)
        st.write("El conteo de nulos por variable se presenta en el ítem 4.")

    with tabs[1]:
        st.subheader("Ítem 2. Clasificación de variables")
        st.write("La función personalizada `clasificar_variables()` separa columnas numéricas y categóricas.")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Variables numéricas", len(numericas))
            st.write(numericas if numericas else "No se detectaron variables numéricas.")
        with c2:
            st.metric("Variables categóricas", len(categoricas))
            st.write(categoricas if categoricas else "No se detectaron variables categóricas.")
        resumen_tipos = pd.DataFrame({
            "variable": datos.columns,
            "tipo_detectado": ["Numérica" if c in numericas else "Categórica" for c in datos.columns]
        })
        st.dataframe(resumen_tipos, use_container_width=True)

    with tabs[2]:
        st.subheader("Ítem 3. Estadísticas descriptivas")
        st.write("Se muestran medidas de tendencia central y dispersión para las columnas disponibles.")
        st.dataframe(datos.describe(include="all").T, use_container_width=True)
        if numericas:
            var = st.selectbox("Variable numérica para interpretar", numericas, key="desc_num")
            serie = pd.to_numeric(datos[var], errors="coerce").dropna()
            if not serie.empty:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Media", f"{serie.mean():,.2f}")
                m2.metric("Mediana", f"{serie.median():,.2f}")
                m3.metric("Desv. estándar", f"{serie.std():,.2f}")
                m4.metric("Moda", f"{analyzer.moda(var)}")
                st.caption(interpretar_distribucion(serie, var))
        else:
            st.info("No hay variables numéricas para calcular estas medidas.")

    with tabs[3]:
        st.subheader("Ítem 4. Análisis de valores faltantes")
        nulos = analyzer.resumen_nulos()
        st.dataframe(nulos, use_container_width=True)
        nulos_graf = nulos[nulos["valores_nulos"] > 0]
        if not nulos_graf.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(data=nulos_graf.reset_index(), x="valores_nulos", y="index", ax=ax, color="#e59b35")
            ax.set_xlabel("Cantidad de valores nulos")
            ax.set_ylabel("Variable")
            ax.set_title("Valores faltantes por variable")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.write("Las variables con mayor cantidad de nulos requieren revisión antes de otros análisis.")
        else:
            st.success("No se detectaron valores nulos explícitos en los registros filtrados.")
            st.caption("Los valores como 'unknown' son categorías del dataset y no necesariamente nulos de Pandas.")

    with tabs[4]:
        st.subheader("Ítem 5. Distribución de variables numéricas")
        if numericas:
            var_hist = st.selectbox("Selecciona una variable numérica", numericas, key="hist_num")
            fig = grafico_histograma(datos, var_hist)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(interpretar_distribucion(datos[var_hist], var_hist))
        else:
            st.info("No hay variables numéricas disponibles.")

    with tabs[5]:
        st.subheader("Ítem 6. Análisis de variables categóricas")
        if categoricas:
            var_cat = st.selectbox("Selecciona una variable categórica", categoricas, key="cat_var")
            conteo = datos[var_cat].astype("string").fillna("Nulo").value_counts()
            tabla = pd.DataFrame({
                "frecuencia": conteo,
                "proporción (%)": (conteo / max(conteo.sum(), 1) * 100).round(2)
            })
            c1, c2 = st.columns([1, 1.3])
            with c1:
                st.dataframe(tabla, use_container_width=True)
            with c2:
                fig = grafico_barras_conteo(datos, var_cat)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No hay variables categóricas disponibles.")

    with tabs[6]:
        st.subheader("Ítem 7. Análisis bivariado: numérica vs resultado")
        if "y" in datos.columns and numericas:
            posibles = [c for c in ["age", "duration", "campaign", "previous"] if c in numericas]
            variable = st.selectbox("Variable numérica", posibles or numericas, key="biv_num")
            resumen = datos.groupby("y", dropna=False)[variable].agg(["count", "mean", "median"]).reset_index()
            st.dataframe(resumen, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(data=datos, x="y", y=variable, ax=ax, color="#82b1e8")
            ax.set_title(f"{variable} por resultado de campaña")
            ax.set_xlabel("Aceptó la campaña (y)")
            ax.set_ylabel(variable)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Compara medianas y dispersión entre grupos; la asociación observada no demuestra causalidad.")
        else:
            st.info("Se necesitan una variable numérica y la columna y para este análisis.")

    with tabs[7]:
        st.subheader("Ítem 8. Análisis bivariado: categórica vs resultado")
        cats_biv = [c for c in categoricas if c != "y"]
        if "y" in datos.columns and cats_biv:
            variable_cat = st.selectbox(
                "Variable categórica",
                [c for c in ["education", "contact", "job", "marital", "month"] if c in cats_biv] or cats_biv,
                key="biv_cat"
            )
            tabla = pd.crosstab(datos[variable_cat], datos["y"], normalize="index") * 100
            st.write("Porcentaje de aceptación/rechazo dentro de cada categoría:")
            st.dataframe(tabla.round(2), use_container_width=True)
            fig, ax = plt.subplots(figsize=(9, 5))
            tabla.plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
            ax.set_title(f"Proporción del resultado por {variable_cat}")
            ax.set_xlabel(variable_cat)
            ax.set_ylabel("Porcentaje")
            ax.legend(title="Resultado")
            plt.xticks(rotation=35, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Se necesitan una variable categórica y la columna y.")

    with tabs[8]:
        st.subheader("Ítem 9. Análisis dinámico según parámetros")
        col1, col2 = st.columns(2)
        with col1:
            variable_x = st.selectbox("Variable para agrupar", datos.columns.tolist(), key="dyn_x")
        with col2:
            variables_ver = st.multiselect(
                "Variables que deseas visualizar",
                options=datos.columns.tolist(),
                default=[c for c in ["age", "duration", "y"] if c in datos.columns],
                key="dyn_multi"
            )
        if variables_ver:
            st.dataframe(datos[[variable_x] + [c for c in variables_ver if c != variable_x]].head(100),
                         use_container_width=True)
        if variable_x in numericas:
            st.write("Resumen numérico por intervalos/categorías:")
            st.dataframe(datos.groupby(variable_x, dropna=False)[numericas].mean(numeric_only=True).head(30),
                         use_container_width=True)
        else:
            conteo_dinamico = datos[variable_x].astype("string").fillna("Nulo").value_counts().head(30)
            st.bar_chart(conteo_dinamico)

    with tabs[9]:
        st.subheader("Ítem 10. Hallazgos clave")
        st.write("Resumen descriptivo calculado con los datos y filtros actuales.")
        if "y" in datos.columns:
            tasa = analyzer.tasa_aceptacion()
            st.metric("Tasa de aceptación del subconjunto", f"{tasa:.2f}%")
            y_counts = datos["y"].astype(str).str.lower().str.strip().value_counts()
            st.write(f"Resultado más frecuente: **{y_counts.index[0]}** ({y_counts.iloc[0]:,} registros).")
        if numericas:
            dispersion = datos[numericas].std(numeric_only=True).dropna().sort_values(ascending=False)
            if not dispersion.empty:
                st.write(f"Mayor desviación estándar entre variables numéricas: **{dispersion.index[0]}** "
                         f"({dispersion.iloc[0]:,.2f}).")
            if "duration" in datos.columns and "y" in datos.columns:
                dur = datos.groupby("y")["duration"].median()
                if len(dur) > 1:
                    st.write("Mediana de duración del contacto por resultado:")
                    st.dataframe(dur.rename("mediana_segundos").to_frame())
        if categoricas:
            st.write("Categoría más frecuente de algunas variables:")
            filas = []
            for c in categoricas[:8]:
                moda = datos[c].mode(dropna=True)
                filas.append({"variable": c, "categoría_más_frecuente": moda.iloc[0] if not moda.empty else "Sin datos"})
            st.dataframe(pd.DataFrame(filas), use_container_width=True)
        st.info("Los hallazgos son descriptivos y deben validarse con contexto comercial antes de tomar decisiones.")

elif pagina == "Conclusiones":
    if "bankmarketing_df" not in st.session_state:
        st.warning("Primero carga el dataset desde «Carga del dataset».")
        st.stop()
    df = st.session_state["bankmarketing_df"].copy()
    st.header("Conclusiones y recomendaciones")
    st.write("Las siguientes conclusiones se generan a partir de los registros cargados. "
             "Revisa los resultados antes de convertirlos en decisiones comerciales.")
    conclusiones = []
    if "y" in df.columns:
        y = df["y"].astype(str).str.lower().str.strip()
        tasa = (y == "yes").mean() * 100
        conclusiones.append(f"La tasa global de aceptación observada en el dataset es {tasa:.2f}%. "
                            "Debe compararse con el indicador histórico de la institución usando la misma definición.")
        conteos = y.value_counts()
        if not conteos.empty:
            conclusiones.append(f"El resultado más frecuente es '{conteos.index[0]}', con {conteos.iloc[0]:,} registros; "
                                "esto permite dimensionar el desbalance entre aceptación y rechazo.")
    if "duration" in df.columns and "y" in df.columns:
        med = df.groupby(df["y"].astype(str).str.lower().str.strip())["duration"].median()
        if "yes" in med.index and "no" in med.index:
            conclusiones.append(f"La mediana de duración del contacto es {med['yes']:.1f} segundos para aceptación "
                                f"y {med['no']:.1f} segundos para rechazo. Es una asociación descriptiva, no una causa.")
    if "contact" in df.columns and "y" in df.columns:
        tasa_contacto = pd.crosstab(df["contact"], df["y"].astype(str).str.lower().str.strip(), normalize="index") * 100
        if "yes" in tasa_contacto.columns and not tasa_contacto.empty:
            mejor = tasa_contacto["yes"].idxmax()
            conclusiones.append(f"Entre los canales observados, '{mejor}' presenta la mayor proporción descriptiva de "
                                f"aceptación ({tasa_contacto.loc[mejor, 'yes']:.2f}%). Revisar tamaño de muestra y costes.")
    nulos = df.isna().sum()
    if nulos.sum() == 0:
        conclusiones.append("No hay valores nulos reconocidos por Pandas; conviene distinguirlos de etiquetas como "
                            "'unknown', que pueden representar información no especificada.")
    else:
        col_nulos = nulos.idxmax()
        conclusiones.append(f"La variable '{col_nulos}' contiene la mayor cantidad de valores nulos "
                            f"({int(nulos[col_nulos])}); debe evaluarse su tratamiento antes de análisis posteriores.")
    while len(conclusiones) < 5:
        conclusiones.append("La distribución de variables demográficas y de contacto debe revisarse por segmento "
                            "para orientar futuras pruebas comerciales sin asumir causalidad.")
    for i, conclusion in enumerate(conclusiones[:5], start=1):
        st.markdown(f"**Conclusión {i}.** {conclusion}")
    st.subheader("Recomendaciones de gestión")
    st.markdown("""
    - Priorizar el seguimiento de los segmentos con mejores tasas observadas, validando que tengan una muestra suficiente.
    - Revisar el coste y la calidad de los canales de contacto antes de reasignar recursos.
    - Evaluar la frecuencia de contactos para evitar gestiones repetitivas poco efectivas.
    - Registrar de forma consistente los resultados y la información de los clientes.
    - Repetir el análisis periódicamente y comparar campañas con métricas definidas de la misma manera.
    """)
    st.caption("Estas recomendaciones no son predicciones ni garantizan resultados; son líneas de revisión derivadas del EDA.")
