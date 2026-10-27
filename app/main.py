"""DiaOps API: serves the diabetes screening risk score.

The output is a screening risk score, not a diagnosis.

The model file is copied into the Docker image at build time, so a running
container always serves exactly the model it was built with.

Settings come from environment variables so the same image works locally and on EC2:
    MODEL_PATH     default models/model.json
    METRICS_PATH   default metrics.json   (shown by /model-info)
    PARAMS_PATH    default params.yaml    (decision threshold)
    MODEL_VERSION  default "dev"          (set to the release tag at build time)
    CORS_ORIGINS   default "*"            (comma-separated; the S3 website URL in production)
"""

import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ConfigDict, Field, create_model

from src.features import FEATURE_RANGES, FEATURES

DISCLAIMER = "Screening risk score only, not a diagnosis. Consult a healthcare professional."

# Request schema built from the same ranges the validate stage checks, so the API
# can never accept a value the model was not trained on. Every field is required,
# and unknown fields are rejected.
PatientFeatures = create_model(
    "PatientFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (int, Field(ge=FEATURE_RANGES[name][0], le=FEATURE_RANGES[name][1]))
        for name in FEATURES
    },
)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "models/model.json"))
    metrics_path = Path(os.getenv("METRICS_PATH", "metrics.json"))
    params_path = Path(os.getenv("PARAMS_PATH", "params.yaml"))

    booster = xgb.Booster()
    booster.load_model(model_path)
    state["booster"] = booster
    # Same md5 that DVC records for models/model.json: ties the running model to dvc.lock.
    state["model_md5"] = hashlib.md5(model_path.read_bytes()).hexdigest()
    state["threshold"] = yaml.safe_load(params_path.read_text())["evaluate"]["threshold"]
    state["metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    yield
    state.clear()


app = FastAPI(title="DiaOps", description=DISCLAIMER, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_version": os.getenv("MODEL_VERSION", "dev"),
        "model_md5": state["model_md5"],
        "threshold": state["threshold"],
        "features": FEATURES,
        "test_metrics": state["metrics"],
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    # Column order and names must match training; DMatrix checks names against the model.
    row = np.array([[getattr(patient, f) for f in FEATURES]], dtype=float)
    risk_score = float(state["booster"].predict(xgb.DMatrix(row, feature_names=FEATURES))[0])
    return {
        "risk_score": round(risk_score, 4),
        "risk_level": "elevated" if risk_score >= state["threshold"] else "low",
        "threshold": state["threshold"],
        "model_version": os.getenv("MODEL_VERSION", "dev"),
        "disclaimer": DISCLAIMER,
    }
