"""Formato de entrada e saida da API.

As categorias sao as mesmas do dataset Telco. Um valor fora delas volta 422 na hora,
em vez de virar uma predicao silenciosa com uma categoria que o modelo nunca viu.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

# Mesmo cliente do exemplo do README, aparece pronto no Swagger.
CUSTOMER_EXAMPLE = {
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


class Customer(BaseModel):
    """Dados de um cliente, com os mesmos nomes de coluna do dataset."""

    # Campos extras (ex.: customerID) sao ignorados, como ja acontecia antes.
    model_config = ConfigDict(extra="ignore", json_schema_extra={"examples": [CUSTOMER_EXAMPLE]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Meses de contrato")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    # Vem vazio no dataset para cliente recem-chegado. O pipeline preenche com a mediana.
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionRequest(BaseModel):
    """Payload do /predict."""

    customer: Customer


class PredictionResponse(BaseModel):
    """Probabilidade de churn, classe prevista e o threshold usado."""

    churn_probability: float
    churn_prediction: int
    threshold: float
