import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Propensión de Compra", layout="centered")

# --- Cargar modelo ---
MODEL_PATH = os.path.join("models", "modelo_propension.joblib")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset_propension_processed.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(PROCESSED_DATA_PATH)
FEATURE_COLUMNS = df.drop(columns=["Target_Compra", "Cliente_ID"]).columns.tolist()

st.title("📊 Modelo de Propensión de Compra")
st.markdown("Ingrese los datos del cliente para predecir la probabilidad de compra.")

# --- Entrada de datos ---
edad = st.number_input("Edad", min_value=18, max_value=90, value=40)
tiene_soat = st.selectbox("¿Tiene SOAT?", [0, 1])
tiene_vida = st.selectbox("¿Tiene Seguro de Vida?", [0, 1])
cantidad_ramos = st.number_input("Cantidad de ramos contratados", min_value=0, max_value=6, value=1)
meses_ultima_compra = st.number_input("Meses desde última compra", min_value=0, max_value=120, value=3)

ciudad = st.selectbox("Ciudad", ["Arequipa", "Cusco", "Lima", "Piura", "Trujillo"])

# --- Crear diccionario para variables dummy ---
ciudades_dummies = {f"Ciudad_{c}": 0 for c in ["Arequipa", "Cusco", "Lima", "Piura", "Trujillo"]}
ciudades_dummies[f"Ciudad_{ciudad}"] = 1

# --- Crear DataFrame para el modelo ---
cliente_data = {
    "Edad": edad,
    "Tiene_SOAT": tiene_soat,
    "Tiene_Vida": tiene_vida,
    "Cantidad_Ramos": cantidad_ramos,
    "Meses_Desde_Ultima_Compra": meses_ultima_compra,
}
cliente_data.update(ciudades_dummies)

df_cliente = pd.DataFrame([cliente_data], columns=FEATURE_COLUMNS)

# --- Predicción ---
if st.button("🔍 Predecir"):
    prob = model.predict_proba(df_cliente)[0][1]
    pred = model.predict(df_cliente)[0]

    st.subheader("📈 Resultados")
    st.write(f"**Probabilidad de compra:** {prob:.2%}")
    st.write(f"**Predicción final:** {'Compra (1)' if pred == 1 else 'No Compra (0)'}")

    if pred == 1:
        st.success("✅ Este cliente tiene alta probabilidad de compra.")
    else:
        st.warning("⚠️ Este cliente tiene baja probabilidad de compra.")
