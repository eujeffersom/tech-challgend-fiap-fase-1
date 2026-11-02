"""API FastAPI para servir predicoes de churn em tempo real."""

import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from sklearn.pipeline import Pipeline
from starlette.middleware.base import BaseHTTPMiddleware

from churn.api_models import PredictionRequest, PredictionResponse
from churn.config import CHAMPION_MODEL_PATH
from churn.logging_config import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Churn Prediction API", version="0.2.0")

# Guarda o campeao em memoria. A chave inclui a data de modificacao do arquivo,
# entao se o modelo for retreinado a API pega a versao nova sem precisar reiniciar.
_champion_cache: dict[str, object] = {"key": None, "artifact": None}


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Registra a latencia de cada requisicao no log."""

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


def _load_champion(path: Path | None = None) -> tuple[Pipeline, float]:
    # O campeao e gerado pelo churn-compare-models. Sem ele nao ha o que servir.
    path = path or CHAMPION_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Modelo campeao nao encontrado. Rode churn-compare-models antes de subir a API."
        )

    key = (str(path), path.stat().st_mtime_ns)
    if _champion_cache["key"] != key:
        _champion_cache["artifact"] = joblib.load(path)
        _champion_cache["key"] = key
        logger.info("champion_loaded", path=str(path))

    artifact = _champion_cache["artifact"]
    return artifact["pipeline"], float(artifact["threshold"])


@app.get("/health")
def health() -> dict[str, str]:
    """Health check usado pelo Docker e pelo deploy."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Roda o pipeline campeao e aplica o threshold escolhido na validacao cruzada."""

    try:
        pipeline, threshold = _load_champion(CHAMPION_MODEL_PATH)
    except FileNotFoundError as exc:
        logger.info("prediction_failed_missing_artifact", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([request.customer.model_dump()])
    # O Pydantic ja garante todos os campos. Isso so pega o caso de um modelo
    # treinado com colunas que a API nao conhece.
    expected_columns = list(getattr(pipeline, "feature_names_in_", row.columns))
    missing = set(expected_columns) - set(row.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Colunas ausentes no payload: {sorted(missing)}",
        )

    probability = float(pipeline.predict_proba(row[expected_columns])[0, 1])
    prediction = int(probability >= threshold)
    logger.info("prediction_served", churn_probability=probability, prediction=prediction)
    return PredictionResponse(
        churn_probability=probability,
        churn_prediction=prediction,
        threshold=threshold,
    )
