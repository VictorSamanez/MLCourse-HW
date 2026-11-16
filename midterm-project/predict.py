"""FastAPI service that serves the bank-term-deposit model."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from pipeline import engineer_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model.bin"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", BASE_DIR / "model_metadata.json"))

with MODEL_PATH.open("rb") as model_file:
    pipeline = pickle.load(model_file)

MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
METADATA: dict[str, Any] = (
    json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if METADATA_PATH.exists()
    else {}
)
THRESHOLD = float(METADATA.get("threshold", 0.5))

app = FastAPI(
    title="Bank Term Deposit Prediction",
    description="Predicts whether a client will subscribe to a term deposit.",
    version="1.0.0",
)


class Client(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[41])
    job: str = Field(..., examples=["technician"])
    marital: str = Field(..., examples=["married"])
    education: str = Field(..., examples=["secondary"])
    default: str = Field(..., examples=["no"])
    balance: int = Field(..., examples=[1200])
    housing: str = Field(..., examples=["yes"])
    loan: str = Field(..., examples=["no"])
    contact: str = Field(..., examples=["cellular"])
    day: int = Field(..., ge=1, le=31, examples=[15])
    month: str = Field(..., examples=["may"])
    campaign: int = Field(..., ge=1, examples=[2])
    pdays: int = Field(..., examples=[-1], description="-1 means never contacted")
    previous: int = Field(..., ge=0, examples=[0])
    poutcome: str = Field(..., examples=["unknown"])


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "bank-term-deposit-prediction",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_sha256": MODEL_SHA256,
        "threshold": THRESHOLD,
        "holdout_metrics": METADATA.get("holdout_metrics", {}),
    }


@app.post("/predict")
def predict(client: Client) -> dict[str, Any]:
    record = pd.DataFrame([client.model_dump()])
    features = engineer_features(record)
    probability = float(pipeline.predict_proba(features)[0, 1])
    return {
        "subscription_probability": round(probability, 6),
        "subscription": bool(probability >= THRESHOLD),
        "threshold": THRESHOLD,
    }
