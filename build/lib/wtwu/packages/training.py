from data import create_global_dataset, create_batches, validate_patient_data
from models import create_model, train_model, save_model_local
from storage import save_metrics_to_bigquery
from models import save_metrics_local
import numpy as np
from wtwu import params
import os

# Configurations
project_id = params.PROJECT_ID
dataset_id = "wtwa_data"
bucket_name = "data-wtwa"
prefix = "preprocessed"
batch_size = 32
patients = [str(i) for i in range(284, 385)]  # Liste des patients (à ajuster selon vos données)

print("Current working directory:", os.getcwd())

# Étape 1 : Valider les données
if not validate_patient_data(bucket_name, prefix, patients):
    raise ValueError("Certaines données patient manquent ou sont incomplètes.")

# Étape 2 : Charger les données globales
print("Création du dataset global...")
all_time_splits, all_labels = create_global_dataset(bucket_name, prefix, patients)

# Étape 3 : Créer les batchs
print("Création des batchs...")
X_batches, y_batches = create_batches(all_time_splits, all_labels, batch_size)

# Étape 4 : Diviser les données en Entraînement, Validation, et Test
print("Division des données en Entraînement, Validation, et Test...")
X_train, y_train = np.concatenate(X_batches[:-2]), np.concatenate(y_batches[:-2])
X_val, y_val = X_batches[-2], y_batches[-2]
X_test, y_test = X_batches[-1], y_batches[-1]

# Étape 5 : Créer le modèle
print("Création du modèle...")
input_shape = X_train.shape[1:]  # Shape des features (1500, 8)
model = create_model(input_shape)

# Étape 6 : Entraîner le modèle
print("Entraînement du modèle...")
history = train_model(model, X_train, y_train, X_val, y_val)

# Étape 7 : Sauvegarder le modèle localement
model_path = save_model_local(model)
print(f"Modèle sauvegardé dans : {model_path}")

# Étape 8 : Collecter et sauvegarder les métriques
print("Sauvegarde des métriques...")
metrics = [
    {
        "epoch": epoch + 1,
        "batch_size": len(y_train),
        "training_loss": history.history['loss'][epoch],
        "validation_loss": history.history['val_loss'][epoch],
        "training_accuracy": history.history['accuracy'][epoch],
        "validation_accuracy": history.history['val_accuracy'][epoch],
        "model_path": model_path
    }
    for epoch in range(len(history.history['loss']))
]

# Sauvegarder les métriques localement
metrics_path = save_metrics_local(metrics)
print(f"Métriques sauvegardées dans : {metrics_path}")

# Optionnel : Sauvegarder les métriques dans BigQuery
save_metrics_to_bigquery(
    project_id=project_id,
    dataset_id=dataset_id,
    table_id="training_metrics",
    metrics=metrics
)
