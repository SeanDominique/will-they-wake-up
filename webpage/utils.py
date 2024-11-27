import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Exemple de fonction pour traiter un fichier uploadé
def process_uploaded_file(file):
    # Ajouter ici la logique pour lire et analyser le fichier
    st.write(f"Traitement du fichier : {file.name}")
    # Exemple de message
    st.info("Le fichier a été traité avec succès.")

# Exemple de fonction pour afficher un graphique
def plot_example_graph():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)
