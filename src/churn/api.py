from typing import Literal

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.config import MLP_MODEL_PATH, PREPROCESSOR_PATH
from churn.features import validate_feature_frame
from churn.logging_config import get_logger
from churn.model import ChurnMLP

logger = get_logger(__name__)
app = FastAPI(title="Churn Prediction API", version="0.1.0")


class CustomerPayload(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionRequest(BaseModel):
    customer: CustomerPayload


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float


def _load_artifacts() -> tuple[object, ChurnMLP, float]:
    if not PREPROCESSOR_PATH.exists() or not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Artefatos nao encontrados. Execute o treino antes de iniciar a API."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    checkpoint = torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=False)
    model = ChurnMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return preprocessor, model, float(checkpoint.get("threshold", 0.5))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        preprocessor, model, threshold = _load_artifacts()
        row = pd.DataFrame([request.customer.model_dump()])
        validate_feature_frame(row)
        features = preprocessor.transform(row)
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32)
            probability = float(torch.sigmoid(model(input_tensor)).item())
        prediction = int(probability >= threshold)
        logger.info("prediction_served", churn_probability=probability, prediction=prediction)
        return PredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            threshold=threshold,
        )
    except FileNotFoundError as exc:
        logger.info("prediction_failed_missing_artifact", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
