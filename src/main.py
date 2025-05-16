from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from typing import Dict
import numpy as np
import pandas as pd
import joblib
import os

#creation d'une instance fastapi
app = FastAPI()
# Spécifier le chemin relatif vers le fichier CSV
data_path = os.path.join('..', 'data', 'data_a_tester.csv')

# charger les données
df = pd.read_csv(data_path)

# Configurer le serveur de suivi MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Nom du modèle enregistré
model_name = "LGBM_Optimized_Model"
# Charger le modèle depuis MLflow
try:
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    print("Modèle chargé avec succès.")
except mlflow.exceptions.MlflowException as e:
    print(f"Erreur lors du chargement du modèle: {e}")

class item(BaseModel):
    id:int
    features: Dict[str,float]

# Créer un dictionnaire d'items
items = {idx: Item(id=idx, features=row.drop('TARGET').to_dict()) for idx, row in df.head(20).iterrows()}

@app.get("/")
def index() -> Dict[str, Dict[int, Item]]:
    return {"items": items}

@app.get("/items/{item_id}")
def query_item_by_id(item_id: int) -> Item:
    if item_id not in items:
        raise HTTPException(status_code=404, detail=f"Item with {item_id=} does not exist.")

    return items[item_id]


def read_root():
    return { "message": "Essai API prediction"}

# point de terminaison (API)
@app.post("/predict") #local : https://127.0.0.1:8000/predict

#fonction de prédiction
def predict(data : Item):
    #nouvelles données sur lesquelles on fait la prédiction
    feature_values = list(data.features.values())
    new_data = [feature_values]

    #prédiction
    class_idx = model.predict(new_data)[0]

    return {"prédiction": class_idx}



