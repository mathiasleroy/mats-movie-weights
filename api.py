"""FastAPI wrapper to serve predictions to the browser extension."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mrp.predict import predict

app = FastAPI(title="Movie Rating Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict_movie(imdb_id: str):
    """
    Returns the predicted rating for a given IMDb ID.
    The model loads into RAM on the first request (takes ~5s), 
    then is instant for all subsequent requests.
    """
    result = predict(imdb_id)
    if result:
        return result
    return {"error": "Could not generate prediction"}

@app.get("/")
def health_check():
    return {"status": "online", "message": "API is running. Use /predict?imdb_id=tt1234567"}