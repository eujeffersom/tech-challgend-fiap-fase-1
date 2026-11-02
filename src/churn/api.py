"""API FastAPI para servir predicoes de churn em tempo real."""

import time
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from starlette.middleware.base import BaseHTTPMiddleware

from churn.config import CHAMPION_MODEL_PATH
from churn.features import validate_feature_frame
from churn.logging_config import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Churn Prediction API", version="0.1.0")


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Registra latencia de cada requisicao em log estruturado."""

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
    """Payload esperado pelo endpoint de predicao."""

    customer: dict[str, Any] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Resposta padronizada com probabilidade, classe e threshold."""

    churn_probability: float
    churn_prediction: int
    threshold: float


def _load_champion() -> tuple[Pipeline, float]:
    # O campeao e gerado pelo churn-compare-models. Sem ele nao ha o que servir.
    if not CHAMPION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo campeao nao encontrado. Rode churn-compare-models antes de subir a API."
        )

    artifact = joblib.load(CHAMPION_MODEL_PATH)
    return artifact["pipeline"], float(artifact["threshold"])


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint simples para health check de deploy."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Roda o pipeline campeao e aplica o threshold escolhido na validacao cruzada."""

    try:
        pipeline, threshold = _load_champion()
        row = pd.DataFrame([request.customer])
        validate_feature_frame(row)
        # O pipeline espera as colunas na mesma ordem do treino.
        expected_columns = list(getattr(pipeline, "feature_names_in_", []))
        missing = set(expected_columns) - set(row.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Colunas ausentes no payload: {sorted(missing)}",
            )
        row = row[expected_columns]
        probability = float(pipeline.predict_proba(row)[0, 1])
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
