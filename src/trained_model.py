import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


# Rutas
PROCESSED_DATA_PATH = os.path.join("data","processed","dataset_propension_processed.csv")
MODEL_PATH = os.path.join ("models","modelo_propension.joblib")

def main ():
    # 1. Cargar datos procesados
    print('Cargando datos procesados...')
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(df.columns)

    # 2. Separar X (features) e Y (target)
    X = df.drop(columns=['Target_Compra', 'Cliente_ID'])
    y = df['Target_Compra']

    #3. Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Entrenar modelos de Random Forest
    print("Entrenando modelos...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #5. Evaluar modelo
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    #6. Guardar modelo entrenado
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

if __name__== "__main__":
    main()






