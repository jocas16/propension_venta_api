import streamlit as st
import requests

# -------------------------------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------------------------------
st.set_page_config(
    page_title="Predicción de Propensión de Compra",
    layout="centered"
)

# URL de tu API FastAPI (asegúrate que esté encendida)
API_URL = "http://127.0.0.1:8000/predict"

st.title("📊 Modelo de Propensión de Compra (vía API)")
st.markdown("Ingrese los datos del cliente y consulte la API para predecir la probabilidad de compra.")

# -------------------------------------------------
# ENTRADAS DEL USUARIO
# -------------------------------------------------
edad = st.number_input("Edad", min_value=18, max_value=90, value=40)
tiene_soat = st.selectbox("¿Tiene SOAT?", [0, 1])
tiene_vida = st.selectbox("¿Tiene Seguro de Vida?", [0, 1])
cantidad_ramos = st.number_input("Cantidad de ramos contratados", min_value=0, max_value=6, value=1)
meses_ultima_compra = st.number_input("Meses desde última compra", min_value=0, max_value=120, value=3)
ciudad = st.selectbox("Ciudad", ["Arequipa", "Cusco", "Lima", "Piura", "Trujillo"])

# -------------------------------------------------
# CONVERSIÓN DE VARIABLES A FORMATO API
# -------------------------------------------------
# Creamos variables dummy para las ciudades (todas en 0 excepto la seleccionada)
ciudades_dummies = {
    "Ciudad_Arequipa": 0,
    "Ciudad_Cusco": 0,
    "Ciudad_Lima": 0,
    "Ciudad_Piura": 0,
    "Ciudad_Trujillo": 0
}
ciudades_dummies[f"Ciudad_{ciudad}"] = 1

# Creamos el JSON con todos los datos en el formato que espera la API
cliente_json = {
    "Edad": edad,
    "Tiene_SOAT": tiene_soat,
    "Tiene_Vida": tiene_vida,
    "Cantidad_Ramos": cantidad_ramos,
    "Meses_Desde_Ultima_Compra": meses_ultima_compra,
    **ciudades_dummies
}

# -------------------------------------------------
# BOTÓN PARA HACER LA PREDICCIÓN
# -------------------------------------------------
if st.button("🔍 Consultar predicción"):
    try:
        # Llamada POST a la API
        response = requests.post(API_URL, json=cliente_json)

        if response.status_code == 200:
            result = response.json()
            prob = result["Probabilidad_compra"]
            pred = result["Prediccion"]

            # Mostrar resultados
            st.subheader("📈 Resultados")
            st.write(f"**Probabilidad de compra:** {prob:.2%}")
            st.write(f"**Predicción final:** {'Compra (1)' if pred == 1 else 'No Compra (0)'}")

            if pred == 1:
                st.success("✅ Este cliente tiene alta probabilidad de compra.")
            else:
                st.warning("⚠️ Este cliente tiene baja probabilidad de compra.")

        else:
            st.error(f"Error en la API: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar con la API. Asegúrate de que esté encendida con FastAPI.")
