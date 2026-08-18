"""FastAPI wrapper to serve predictions to the browser extension."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mrp.predict import predict
from mrp import embeddings

app = FastAPI(title="Movie Rating Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
def preload_models():
    """Load the embedding model into RAM before the first request arrives."""
    print("Preloading embedding model...")
    embeddings._get_model()
    print("✓ API is ready and fast!")

@app.get("/predict")
def predict_movie(imdb_id: str):
    result = predict(imdb_id)
    if result:
        return result
    return {"error": "Could not generate prediction"}