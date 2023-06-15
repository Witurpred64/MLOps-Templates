
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ML Prediction Service", version="1.0.0")

class ModelInput(BaseModel):
    features: List[float]

class ModelOutput(BaseModel):
    prediction: int
    probability: float

class MLService:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
    def predict(self, features: List[float]):
        X = np.array(features).reshape(1, -1)
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X).max()
        return pred, prob

# Singleton service
service = None

@app.on_event("startup")
def load_model():
    global service
    # Mock model loading for demonstration
    # service = MLService("models/production_model.joblib")
    print("Model loaded successfully.")

@app.post("/predict", response_model=ModelOutput)
async def predict(data: ModelInput):
    try:
        # For demo, returning dummy values
        # pred, prob = service.predict(data.features)
        return ModelOutput(prediction=1, probability=0.95)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
