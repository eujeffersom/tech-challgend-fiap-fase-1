import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import churn.api as api
from churn.compare_models import save_champion
from churn.data import generate_sample_dataset, normalize_churn_dataframe, split_features_target
from churn.features import build_preprocessor

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
def fake_champion(tmp_path, monkeypatch):
    """Treina um campeao pequeno numa pasta temporaria, sem depender de models/."""

    df = normalize_churn_dataframe(generate_sample_dataset(rows=80))
    x, y = split_features_target(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x)),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    ).fit(x, y)

    champion_path = tmp_path / "champion.joblib"
    save_champion(pipeline, model_name="logistic_regression", threshold=0.5, path=champion_path)
    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", champion_path)


def test_health_endpoint() -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Process-Time-ms" in response.headers


def test_predict_contract(fake_champion) -> None:
    client = TestClient(api.app)
    response = client.post("/predict", json={"customer": CUSTOMER})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "churn_prediction", "threshold"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {0, 1}
    assert body["churn_prediction"] == int(body["churn_probability"] >= body["threshold"])


def test_predict_returns_422_when_columns_are_missing(fake_champion) -> None:
    client = TestClient(api.app)
    response = client.post("/predict", json={"customer": {"gender": "Female"}})
    assert response.status_code == 422


def test_predict_returns_503_without_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", tmp_path / "nao_existe.joblib")
    client = TestClient(api.app)
    response = client.post("/predict", json={"customer": CUSTOMER})
    assert response.status_code == 503
