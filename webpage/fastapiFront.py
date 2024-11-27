import requests
import streamlit as st

# Fonction pour appeler l'API et obtenir une prédiction
def get_prediction(payload):
    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=payload)
        if response.status_code == 200:
            return response.json().get("prediction")
        else:
            st.error(f"Erreur API : {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return None
