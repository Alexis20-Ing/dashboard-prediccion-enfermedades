
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Dashboard Analítico - Predicción de Enfermedades",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CARGA DE DATOS
# =====================================================

@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_personal.csv")

try:
    df = cargar_datos()
except Exception as e:
    st.error("❌ No se pudo cargar dataset_personal.csv")
    st.error(e)
    st.stop()

# =====================================================
# TÍTULO
# =====================================================

st.title("🩺 Dashboard Analítico")
st.subheader("Predicción de Enfermedades mediante Data Engineering y Data Science")

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🔎 Filtros")

grupo = st.sidebar.multiselect(
    "Grupo de Edad",
    sorted(df["GrupoEdad"].unique()),
    default=sorted(df["GrupoEdad"].unique())
)

actividad = st.sidebar.multiselect(
    "Actividad Física",
    sorted(df["ActividadFisica"].unique()),
    default=sorted(df["ActividadFisica"].unique())
)

df = df[
    (df["GrupoEdad"].isin(grupo)) &
    (df["ActividadFisica"].isin(actividad))
]

if len(df) == 0:
    st.warning("No existen registros para los filtros seleccionados.")
    st.stop()

# =====================================================
# KPIs
# =====================================================

total = len(df)

enfermos = int(df["Enfermedad"].sum())

porcentaje = enfermos / total * 100

edad_promedio = df["Edad"].mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric("👥 Pacientes", total)

k2.metric("🩺 Enfermos", enfermos)

k3.metric("% Enfermedad", f"{porcentaje:.2f}%")

k4.metric("Edad Promedio", f"{edad_promedio:.1f}")

st.divider()

# =====================================================
# PRIMERA FILA
# =====================================================

c1, c2 = st.columns(2)

with c1:

    st.subheader("Pacientes por Grupo de Edad")

    barras = (
        df["GrupoEdad"]
        .value_counts()
        .reset_index()
    )

    barras.columns = ["GrupoEdad", "Cantidad"]

    fig = px.bar(
        barras,
        x="GrupoEdad",
        y="Cantidad",
        color="GrupoEdad",
        text="Cantidad"
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

with c2:

    st.subheader("Distribución del IMC")

    fig = px.histogram(
        df,
        x="IMC",
        nbins=20,
        color_discrete_sequence=["royalblue"]
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# SEGUNDA FILA
# =====================================================

c3, c4 = st.columns(2)

with c3:

    st.subheader("IMC vs Glucosa")

    fig = px.scatter(
        df,
        x="IMC",
        y="Glucosa",
        color=df["Enfermedad"].astype(str),
        hover_data=["Edad"],
        labels={"color":"Enfermedad"}
    )

    st.plotly_chart(fig, use_container_width=True)

with c4:

    st.subheader("Pacientes según Actividad Física")

    pastel = (
        df["ActividadFisica"]
        .value_counts()
        .reset_index()
    )

    pastel.columns = ["Actividad", "Cantidad"]

    fig = px.pie(
        pastel,
        names="Actividad",
        values="Cantidad",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HEATMAP
# =====================================================

st.subheader("Mapa de Correlación")

corr = df.copy()

corr["ActividadFisica"] = corr["ActividadFisica"].astype("category").cat.codes
corr["GrupoEdad"] = corr["GrupoEdad"].astype("category").cat.codes
corr["RiesgoMetabolico"] = corr["RiesgoMetabolico"].astype("category").cat.codes

matriz = corr.corr(numeric_only=True)

fig = go.Figure(
    data=go.Heatmap(
        z=matriz.values,
        x=matriz.columns,
        y=matriz.columns,
        colorscale="Blues",
        text=matriz.round(2).values,
        texttemplate="%{text}",
        showscale=True
    )
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# STORYTELLING
# =====================================================

st.divider()

st.header("📋 Storytelling de Datos")

izq, der = st.columns(2)

with izq:

    st.success("Hallazgos Principales")

    st.write("✅ La mayor concentración de pacientes pertenece al grupo Adulto.")

    st.write("✅ Los pacientes con mayor IMC y Glucosa presentan mayor probabilidad de enfermedad.")

    st.write("✅ El modelo Random Forest alcanzó un Accuracy de 98.70%, siendo el mejor modelo predictivo.")

with der:

    st.info("Recomendaciones")

    st.write("✔ Implementar campañas preventivas para pacientes con IMC elevado.")

    st.write("✔ Incrementar controles periódicos de glucosa y presión arterial.")

    st.write("✔ Utilizar el modelo predictivo para priorizar la atención médica.")

st.divider()

st.caption("Proyecto desarrollado con Streamlit • Data Engineering • Data Science")
