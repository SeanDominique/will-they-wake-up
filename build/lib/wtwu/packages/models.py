from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import json

def create_model(input_shape):
    """
    Crée un modèle RNN avec des couches LSTM pour traiter les données temporelles.
    """
    model = Sequential()

    # Première couche LSTM
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))  # Régularisation

    # Deuxième couche LSTM
    model.add(LSTM(32, activation='tanh'))
    model.add(Dropout(0.3))

    # Couche Dense pour la sortie
    model.add(Dense(1, activation='sigmoid'))  # Sortie binaire (0 ou 1)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, save_path="./models/best_model.keras", epochs=20):
    """
    Entraîne le modèle avec les données fournies.
    """
    # Callbacks pour stopper l'entraînement et sauvegarder le meilleur modèle
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True
    )

    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=min(32, len(y_train) // 10),  # Ajuster selon la taille des données
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test.
    """
    # Prédictions
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Évaluation sur l'ensemble de test :")
    print(f"Précision (Accuracy) : {accuracy:.4f}")
    print(f"Précision (Precision) : {precision:.4f}")
    print(f"Rappel (Recall) : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Matrice de confusion :\n{cm}")

    return y_pred, y_pred_prob, accuracy, precision, recall, f1


def save_model_local(model, base_dir="./Metrics_wtwa"):
    """
    Sauvegarde le modèle localement dans un dossier dédié avec un nom incrémenté et une date.
    """
    # Créer le répertoire principal s'il n'existe pas
    os.makedirs(base_dir, exist_ok=True)

    # Générer un nom unique basé sur la date et l'incrémentation
    date_today = datetime.now().strftime("%Y-%m-%d")
    existing_files = [f for f in os.listdir(base_dir) if f.startswith("model_")]
    next_index = len(existing_files) + 1
    save_path = os.path.join(base_dir, f"model_{next_index}_{date_today}.h5")

    # Sauvegarder le modèle
    model.save(save_path)
    print(f"Modèle sauvegardé localement dans {save_path}")
    return save_path


def save_metrics_local(metrics, base_dir="./Metrics_wtwa"):
    """
    Sauvegarde les métriques d'entraînement/validation localement dans un fichier JSON avec un nom unique.
    """
    # Créer le répertoire principal s'il n'existe pas
    os.makedirs(base_dir, exist_ok=True)

    # Générer un nom unique basé sur la date et l'incrémentation
    date_today = datetime.now().strftime("%Y-%m-%d")
    existing_files = [f for f in os.listdir(base_dir) if f.startswith("metrics_")]
    next_index = len(existing_files) + 1
    save_path = os.path.join(base_dir, f"metrics_{next_index}_{date_today}.json")

    # Sauvegarder les métriques dans un fichier JSON
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métriques sauvegardées localement dans {save_path}")
    return save_path
