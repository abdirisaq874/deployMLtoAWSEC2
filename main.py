from fastapi import FastAPI
from pydantic import BaseModel
from App.model.model import predict_pipeline
from mangum import Mangum as mangum
import numpy as np
# import logging

app = FastAPI()

handler = mangum.Handler(app)


class TextIn(BaseModel):
    UV: float
    RGB_B: float
    RGB_G: float
    RGB_R: float
    NIR: float
    SICAKLIK: float
    NEM: float


class PredictionOut(BaseModel):
    bitki_turu: str
    Confidence: float


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "0.0.1"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    bitki_turu, Confidence = predict_pipeline(payload.dict())
    return {"bitki_turu": bitki_turu, "Confidence": Confidence}
