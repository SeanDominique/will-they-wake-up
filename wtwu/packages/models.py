from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from keras_tuner.tuners import RandomSearch


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import json

def true_positive_ratio(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return true_positives/(false_negatives + false_positives)

def false_positive_ratio(y_true,y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return false_positives / (false_positives + true_negatives)

def build_model(hp):
    """
    Crée un modèle Keras avec des hyperparamètres ajustables pour Grid Search.

    Args:
        hp (kerastuner.HyperParameters): Objet pour définir les hyperparamètres.

    Returns:
        tf.keras.Model: Modèle compilé.
    """
    model = Sequential([
        # Couche Bidirectionnelle LSTM
        Bidirectional(LSTM(
            units=hp.Int('lstm_units', min_value=64, max_value=256, step=64),
            return_sequences=True,
            kernel_regularizer=l2(hp.Float('l2_reg', min_value=0.001, max_value=0.01, step=0.002))
        ), input_shape=(1500, 8)),
        Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),

        # Couche LSTM supplémentaire
        LSTM(
            units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32),
            return_sequences=False,
            kernel_regularizer=l2(hp.Float('l2_reg_2', min_value=0.001, max_value=0.01, step=0.002))
        ),
        Dropout(hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)),

        # Couche Dense
        Dense(
            units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu',
            kernel_regularizer=l2(hp.Float('l2_reg_dense', min_value=0.001, max_value=0.01, step=0.002))
        ),

        # Couche de sortie
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='binary_crossentropy',
        metrics=['accuracy', 'recall']
    )
    return model

def optimize_model(X_train, y_train, X_val, y_val, max_trials=10, executions_per_trial=1):
    """
    Optimise les hyperparamètres du modèle avec KerasTuner et Grid Search.

    Args:
        X_train (np.array): Données d'entraînement.
        y_train (np.array): Labels d'entraînement.
        X_val (np.array): Données de validation.
        y_val (np.array): Labels de validation.
        max_trials (int): Nombre de combinaisons d'hyperparamètres à tester.
        executions_per_trial (int): Nombre d'exécutions pour chaque essai.

    Returns:
        tf.keras.Model: Meilleur modèle trouvé.
        dict: Meilleurs hyperparamètres.
    """
    tuner = RandomSearch(
        build_model,
        objective=['val_recall','val_accuracy'],
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='hyperparameter_tuning',
        project_name='rnn_grid_search'
    )

    print("Lancement de la recherche d'hyperparamètres...")
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=32)

    # Obtenir les meilleurs hyperparamètres
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Meilleurs hyperparamètres trouvés : {best_hps.values}")

    # Construire le meilleur modèle
    best_model = tuner.hypermodel.build(best_hps)
    return best_model, best_hps

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
        metrics=['accuracy','recall','precision']
    )

    return model

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=False), input_shape=(1500, 8)),
    Dense(1, activation="sigmoid")
])

def create_model2(input_shape):

    """

    """
    model2 = Sequential()
    #Crée un modèle RNN avec des couches LSTM pour traiter les données temporelles.
    # Première couche LSTM
    model2.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=(1500, 8)))
    model2.Dropout((0.3))
    #deuxieme couche  LSTM
    model2.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape))
    model2.add(Dropout(0.3))  # Régularisation

    # Deuxième couche LSTM
    model2.add(LSTM(32, activation='tanh'))
    model2.add(Dropout(0.3))

    #Troisieme couche LSTM

    # Couche Dense pour la sortie
    model2.add(Dense(1, activation='sigmoid'))  # Sortie binaire (0 ou 1)

    model2.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'recall']
    )

    return model2


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
        shuffle=True,
        batch_size=min(128, len(y_train) // 10),  # Ajuster selon la taille des données
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
    tpr = true_positive_ratio(y_test,y_pred)

    print("Évaluation sur l'ensemble de test :")
    print(f"Précision (Accuracy) : {accuracy:.4f}")
    print(f"Précision (Precision) : {precision:.4f}")
    print(f"Rappel (Recall) : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"True Positive Ratio : {tpr:.4f}")
    print(f"Matrice de confusion :\n{cm}")

    return y_pred, y_pred_prob, accuracy, precision, recall, f1, tpr


def save_model_local(model, base_dir="./Metrics_wtwa", time_window=None,window_size = 24):
    """
    Sauvegarde le modèle localement dans un dossier dédié avec un nom incrémenté et une date.
    """
    # Créer le répertoire principal s'il n'existe pas
    os.makedirs(base_dir, exist_ok=True)

    # Générer un nom unique basé sur la date et l'incrémentation
    if time_window == None:
        date_today = datetime.now().strftime("%Y-%m-%d")
        existing_files = [f for f in os.listdir(base_dir) if f.startswith("model_")]
        next_index = len(existing_files) + 1
        save_path = os.path.join(base_dir, f"model_{next_index}_{date_today}.h5")

        # Sauvegarder le modèle
        model.save(save_path)
        print(f"Modèle sauvegardé localement dans {save_path}")
        return save_path
    else:
        date_today = datetime.now().strftime("%Y-%m-%d")
        existing_files = [f for f in os.listdir(base_dir) if f.startswith("model_")]
        next_index = len(existing_files) + 1
        save_path = os.path.join(base_dir, f"model_{next_index}_{date_today}_{time_window}_to_{time_window + window_size}_hours.h5")

        # Sauvegarder le modèle
        model.save(save_path)
        print(f"Modèle sauvegardé localement dans {save_path}")
        return save_path

def save_metrics_local(metrics, base_dir="./Metrics_wtwa", time_window=None, window_size=24):
    """
    Sauvegarde les métriques d'entraînement/validation localement dans un fichier JSON avec un nom unique.
    """
    # Créer le répertoire principal s'il n'existe pas
    os.makedirs(base_dir, exist_ok=True)
    if time_window == None:
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
    else:
        # Générer un nom unique basé sur la date et l'incrémentation
        date_today = datetime.now().strftime("%Y-%m-%d")
        existing_files = [f for f in os.listdir(base_dir) if f.startswith("metrics_")]
        next_index = len(existing_files) + 1
        save_path = os.path.join(base_dir, f"metrics_{next_index}_{date_today}_{time_window}_to_{time_window + window_size}_hours.json")

        # Sauvegarder les métriques dans un fichier JSON
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Métriques sauvegardées localement dans {save_path}")
        return save_path

def save_best_hps_local_hyperparameters(directory, best_hps, metrics):
    """
    Sauvegarde les meilleurs hyperparamètres et métriques localement dans un dossier dédié.

    Args:
        directory (str): Dossier parent où créer le dossier "Hyperparametres".
        best_hps (dict): Meilleurs hyperparamètres trouvés.
        metrics (dict): Métriques associées au meilleur modèle.
    """
    # Créer le dossier principal et le sous-dossier "Hyperparametres"
    hyperparam_dir = os.path.join(directory, "Hyperparametres")
    os.makedirs(hyperparam_dir, exist_ok=True)

    # Préparer les données
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "hyperparameters": best_hps.values, # Extraction des hyperparamètres
        "best_val_accuracy": metrics['accuracy'],
        "best_val_recall": metrics['recall'],
        "best_val_loss": metrics['loss']
    }

    # Nom du fichier avec timestamp
    filename = os.path.join(hyperparam_dir, f"best_hps_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")

    # Sauvegarder les données dans un fichier JSON
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Meilleurs hyperparamètres et métriques sauvegardés localement dans : {filename}")
    return filename
