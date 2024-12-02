from models import create_model
from storage import save_predictions_to_bigquery
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Configurations
project_id = "your_project_id"
dataset_id = "wtwa_data"
table_id = "predictions"
bucket_name = "data-wtwa"
prefix = "preprocessed"
batch_size = 32
model_path = "./Metrics_wtwa/model_1_2024-11-30.h5"  # Chemin du modèle sauvegardé

# Charger le modèle sauvegardé
model = load_model(model_path)
print(f"Modèle chargé depuis : {model_path}")

# Charger les données de test
# Si vous utilisez les batchs existants
X_batches, y_batches = create_batches(bucket_name, prefix, patients, batch_size)
X_test = X_batches[-1]
y_test = y_batches[-1]

# Faire des prédictions
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype("int32")

# Collecter les prédictions pour BigQuery
predictions = [
    {
        "patient_id": patients[i],
        "true_label": int(y_test[i]),
        "predicted_label": int(y_pred[i]),
        "prediction_prob": float(y_pred_prob[i]),
    }
    for i in range(len(y_test))
]

# Optionnel : Sauvegarder les prédictions dans BigQuery
save_predictions_to_bigquery(
    project_id=project_id,
    dataset_id=dataset_id,
    table_id=table_id,
    predictions=predictions
)

# Sauvegarder localement si nécessaire

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("./Metrics_wtwa/predictions.csv", index=False)
print("Prédictions sauvegardées localement dans ./Metrics_wtwa/predictions.csv")
