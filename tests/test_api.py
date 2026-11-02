import joblib
import pytest
import torch
from fastapi.testclient import TestClient

import churn.api as api
from churn.data import generate_sample_dataset, split_features_target
from churn.features import build_preprocessor
from churn.model import ChurnMLP

# Mesmo cliente do exemplo do README. Se esse payload parar de funcionar,
# quem consome a API quebra junto.
CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.5,
    "TotalCharges": 950.0,
}


@pytest.fixture
def fake_artifacts(tmp_path, monkeypatch):
    """Gera artefatos pequenos numa pasta temporaria, sem depender de models/."""

    df = generate_sample_dataset(rows=60)
    x, _ = split_features_target(df)
    preprocessor = build_preprocessor(x)
    n_features = preprocessor.fit_transform(x).shape[1]

    # Pesos aleatorios bastam: aqui so interessa o formato da resposta.
    model = ChurnMLP(input_dim=n_features)
    preprocessor_path = tmp_path / "preprocessor.joblib"
    model_path = tmp_path / "churn_mlp.pt"
    joblib.dump(preprocessor, preprocessor_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": n_features,
            "hidden_dim": 64,
            "dropout": 0.2,
            "threshold": 0.5,
        },
        model_path,
    )

    monkeypatch.setattr(api, "PREPROCESSOR_PATH", preprocessor_path)
    monkeypatch.setattr(api, "MLP_MODEL_PATH", model_path)


def test_health_endpoint() -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Process-Time-ms" in response.headers


def test_predict_contract(fake_artifacts) -> None:
    client = TestClient(api.app)
    response = client.post("/predict", json={"customer": CUSTOMER})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "churn_prediction", "threshold"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {0, 1}
    assert body["churn_prediction"] == int(body["churn_probability"] >= body["threshold"])


def test_predict_returns_422_when_columns_are_missing(fake_artifacts) -> None:
    client = TestClient(api.app)
    response = client.post("/predict", json={"customer": {"gender": "Female"}})
    assert response.status_code == 422


def test_predict_returns_503_without_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", tmp_path / "nao_existe.joblib")
    monkeypatch.setattr(api, "MLP_MODEL_PATH", tmp_path / "nao_existe.pt")
    client = TestClient(api.app)
    response = client.post("/predict", json={"customer": CUSTOMER})
    assert response.status_code == 503
