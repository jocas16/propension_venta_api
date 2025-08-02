import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Inicializar API

app = FastAPI(
        title="API - Modelo de propension de venta",
        description= "Servicio para predecir probabilidad de compra de un cliente",
        version="1.0"
)

# Rutas del modelo y dataset procesado
MODELO_PATH = os.path.join("models", "modelo_propension.joblib")
PROCESSED_DATA_PATH = os.path.join("data","processed","dataset_propension_processed.csv")

# Cargar modelo y columnas esperadas
model = joblib.load(MODELO_PATH)
df = pd.read_csv(PROCESSED_DATA_PATH)
FEATURE_COLUMNS = df.drop(columns=["Target_Compra","Cliente_ID"]).columns.to_list()

# Clase para recibir datos en JSON
class ClienteInput (BaseModel):
    Edad:       int
    Tiene_SOAT: int
    Tiene_Vida: int
    Cantidad_Ramos: int
    Meses_Desde_Ultima_Compra: int
    Ciudad_Arequipa: int
    Ciudad_Cusco: int
    Ciudad_Lima : int
    Ciudad_Piura: int
    Ciudad_Trujillo: int

@app.post("/predict")

def predict (cliente: ClienteInput):
    # Convertir a dataframe
    data_dict = cliente.dict() 
    df_cliente = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)

    # Predecir 
    prob = model.predict_proba(df_cliente)[0][1]
    pred = model.predict(df_cliente)[0]

    return {
        "Probabilidad_compra": round(float(prob), 4),
        "Prediccion" : int(pred)
    }

# Ruta de prueba
@app.get("/")
def home():
    return {"mensaje": "API de Propension de Venta funcionando correctamente"}























