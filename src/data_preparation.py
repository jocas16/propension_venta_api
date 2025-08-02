import pandas as pd
import os

# Rutas de entrada y salida
RAW_DATA_PATH = os.path.join("data", "raw", "dataset_propension.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset_propension_processed.csv")


def main():
    # 1. Cargar el dataset
    print("Cargando dataset desde:", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    # 2. Revisar si hay nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # 3. One-Hot Encoding para 'Ciudad'
    df_encoded = pd.get_dummies(df, columns=["Ciudad"], drop_first=True)

    # 4. Guardar dataset procesado
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df_encoded.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nDataset procesado guardado en:", PROCESSED_DATA_PATH)
    print("Forma final del dataset:", df_encoded.shape)

if __name__ == "__main__":
    main()
