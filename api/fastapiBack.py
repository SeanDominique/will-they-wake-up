from api.fastapiBack import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurer les origines autorisées
origins = [
    "http://localhost:8501",  # Front-end local Streamlit
    "http://127.0.0.1:8501",  # Une autre variante locale
    "https://mon-site-front-end.com"  # Domaine du front-end déployé
]

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Domaines autorisés
    allow_credentials=True,  # Autoriser les cookies ou les sessions
    allow_methods=["*"],  # Méthodes HTTP autorisées
    allow_headers=["*"],  # En-têtes autorisés
)

model = joblib.load("../wtwu-packages/models/model.pkl")

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict/")
def predict(data: dict):
    features = data["features"]
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
