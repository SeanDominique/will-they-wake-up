from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

def create_model(input_shape, dense_units=32, dropout_rate=0.3):
    """
    model LSTM bidirectionne
    Parametres:
        input_shape (tuple): Dimensions de l'entrée (n_timepoints, n_channels).
        lstm_units (tuple): Neurones pour chaque couche LSTM bidirectionnelle.
        dense_units (int): Neurones dans la couche Fully Connected.
        dropout_rate (float): Taux de Dropout.
    Returns:
        model (Sequential): Modèle Keras compilé.
    """
    model = Sequential([
        # couche LSTM
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
        Dropout(dropout_rate),

        # couche LSTM
        Bidirectional(LSTM(32)),
        Dropout(dropout_rate),

        # Couche Fully Connected
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),

        # Couche de sortie pour classification binaire
        Dense(1, activation='sigmoid')
    ])

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
    """
    Compile le modèle avec les paramètres spécifiés.

    Parameters:
        model (Sequential): Le modèle Keras à compiler.
        optimizer (str or keras.optimizers): Optimisateur à utiliser (par défaut 'adam').
        loss (str): Fonction de perte (par défaut 'binary_crossentropy').
        metrics (list): Liste des métriques à suivre (par défaut ['accuracy']).

    Returns:
        model (Sequential): Modèle Keras compilé.
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=20, use_early_stopping=False):
    """
    Entraîne le modèle avec possibilité d'utiliser l'Early Stopping.

    Parameters:
        model (Sequential): Modèle Keras à entraîner.
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Labels d'entraînement.
        X_val (ndarray): Données de validation (optionnel).
        y_val (ndarray): Labels de validation (optionnel).
        batch_size (int): Taille des lots (par défaut 32).
        epochs (int): Nombre d'époques (par défaut 20).
        use_early_stopping (bool): Utiliser l'Early Stopping (par défaut False).

    Returns:
        history (History): Historique d'entraînement du modèle.
    """
    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

    # Entraînement avec ou sans validation
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            early_stopping=early_stopping
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping=early_stopping
        )
    return history


def evaluate_model(model, X_test, y_test, batch_size=8):
    """
    Évalue le modèle sur les données de test.
    Parameters:
        model (Sequential): Modèle Keras à évaluer.
        X_test (ndarray): Données de test.
        y_test (ndarray): Labels de test.
        batch_size (int): Taille des lots (par défaut 32)
    Returns:
        results (dict): Dictionnaire contenant la perte et les métriques spécifiées.
    """

    # Évaluation du modèle
    evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    # Récupérer les noms des métriques
    metrics_names = model.metrics_names
    # Créer un dictionnaire des résultats
    results = {metric: value for metric, value in zip(metrics_names, evaluation)}

    return results


def model_predict():
    # get new data
    # from storage import *

    # clean, preprocess the data
    # from preprocess import *

    # TODO: actually make the prediction

    # save the scores of the model in GCS
    # from model import *

    pass
s
create_model(dense_units=32, dropout_rate=0.3)
