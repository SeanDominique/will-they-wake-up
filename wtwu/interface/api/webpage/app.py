import streamlit as st
from fastapiFront import get_prediction
from utils import process_uploaded_file, plot_example_graph

st.set_page_config(page_title="Prédictions Médicales", layout="centered")

st.title("Interface Médicale - Prédictions")

st.header("Entrez les caractéristiques du patient")
feature1 = st.number_input("Caractéristique 1")
feature2 = st.number_input("Caractéristique 2")
feature3 = st.number_input("Caractéristique 3")


if st.button("Faire une prédiction"):
    payload = {"features": [feature1, feature2, feature3]}
    prediction = get_prediction(payload)  # Appelle l'API via api_client.py

    if prediction is not None:
        st.success(f"Résultat de la prédiction : {prediction}")
        if prediction == 0:
            st.warning("Le modèle indique une issue non favorable. Consultez l'équipe médicale.")
        else:
            st.info("Prédiction favorable.")

st.header("Visualisation des données")
st.write("Exemple de graphique basé sur des données simulées.")
plot_example_graph()  #graphe EEG ECG

# --- Section : Upload de fichier ---
st.header("Uploader un fichier")
uploaded_file = st.file_uploader("Téléchargez un fichier (CSV ou MAT)", type=["csv", "mat"])
if uploaded_file:
    st.write("Fichier chargé avec succès.")
    process_uploaded_file(uploaded_file)  # Traite le fichier uploadé
