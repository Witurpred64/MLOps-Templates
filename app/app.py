
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    # In a real scenario, load your model here and make a prediction
    prediction = sum(request.data) / len(request.data) # Example: simple average
    return {"prediction": prediction}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
