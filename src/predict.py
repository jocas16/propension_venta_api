import pandas as pd
import joblib
import os

# Ruta del modelo entrenado y dataset procesado

MODEL_PATH = os.path.join("models","modelo_propension.joblib")
PROCESSED_DATA_PATH =  os.path.join("data","processed","dataset_propension_processed.csv")

def main ():
    #1. Cargar modelo
    model = joblib.load(MODEL_PATH)

    #2. Cargar dataset procesado para obtener estructura de columnas
    df = pd.read_csv(PROCESSED_DATA_PATH)
    features_columns = df.drop(columns=["Target_Compra", "Cliente_ID"]).columns

    #3. Crear un cliente ficticio (ajuste valores)
    nuevo_cliente = pd.DataFrame([{
        "Edad": 40,
        "Tiene_SOAT":1,
        "Tiene_Vida":0,
        "Cantidad_Ramos":1,
        "Meses_Desde_Ultima_Compra":3,
        # Variables dummy de ciudad
        "Ciudad_Arequipa":0,
        "Ciudad_Cusco"   :0,
        "Ciudad_Lima"    :1,
        "Ciudad_Piura"   :0,
        "Ciudad_Trujillo":0       
    }], columns=features_columns)

    #4. Predecir la probabilidad
    prob = model.predict_proba(nuevo_cliente)[0][1]
    pred = model.predict(nuevo_cliente)[0]

    print(f"Probabilidad de compra:  {prob:.2f}")
    print(f"Prediccion final (1=compra, 0 = No compra ): {pred}")

if __name__ == "__main__":
    main()



















