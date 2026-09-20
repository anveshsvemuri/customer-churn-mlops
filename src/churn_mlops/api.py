from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES
from .model import load_model, predict


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=500)
    support_tickets: int = Field(ge=0, le=50)
    contract_months: int = Field(ge=1, le=36)
    autopay: int = Field(ge=0, le=1)
    streaming_services: int = Field(ge=0, le=10)


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_version: str = "0.1.0"


@lru_cache
def get_model():
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    if not path.exists():
        raise FileNotFoundError("Train the model first: churn-mlops train")
    return load_model(path)


app = FastAPI(title="Customer Churn MLOps API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict_customer(customer: CustomerFeatures) -> Prediction:
    try:
        frame = pd.DataFrame([customer.model_dump()], columns=FEATURES)
        scored = predict(get_model(), frame).iloc[0]
        return Prediction(
            churn_probability=float(scored["churn_probability"]),
            churn_prediction=int(scored["churn_prediction"]),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

