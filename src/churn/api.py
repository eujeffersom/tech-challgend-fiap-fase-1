import time
from typing import Any

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from churn.config import MLP_MODEL_PATH, PREPROCESSOR_PATH
from churn.features import validate_feature_frame
from churn.logging_config import get_logger
from churn.model import ChurnMLP

logger = get_logger(__name__)
app = FastAPI(title="Churn Prediction API", version="0.1.0")


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response


app.add_middleware(LatencyLoggingMiddleware)


class PredictionRequest(BaseModel):
    customer: dict[str, Any] = Field(min_length=1)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float


def _load_artifacts() -> tuple[object, ChurnMLP, float]:
    # A API depende dos artefatos gerados pelo treino da MLP.
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
        row = pd.DataFrame([request.customer])
        validate_feature_frame(row)
        # Garante que a entrada siga a mesma ordem de colunas usada no treino.
        expected_columns = list(getattr(preprocessor, "feature_names_in_", []))
        missing = set(expected_columns) - set(row.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Colunas ausentes no payload: {sorted(missing)}",
            )
        row = row[expected_columns]
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
