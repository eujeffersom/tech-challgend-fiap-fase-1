"""Schema Pandera para validar o CSV bruto Telco Customer Churn."""

import pandera.pandas as pa

from churn.config import TARGET_ALIASES

TELCO_CHURN_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": pa.Column(int, pa.Check.ge(0)),
        "PhoneService": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": pa.Column(str),
        "InternetService": pa.Column(str),
        "OnlineSecurity": pa.Column(str),
        "OnlineBackup": pa.Column(str),
        "DeviceProtection": pa.Column(str),
        "TechSupport": pa.Column(str),
        "StreamingTV": pa.Column(str),
        "StreamingMovies": pa.Column(str),
        "Contract": pa.Column(str),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": pa.Column(str),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0), coerce=True),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
    },
    coerce=True,
)


def validate_raw_churn_schema(df):
    """Valida o dataset bruto aceitando aliases conhecidos da coluna alvo."""

    target_column = next((column for column in TARGET_ALIASES if column in df.columns), None)
    if target_column is None:
        raise ValueError(f"Coluna alvo ausente. Use uma destas colunas: {TARGET_ALIASES}")
    normalized = df.rename(columns={target_column: "Churn"})
    return TELCO_CHURN_SCHEMA.validate(normalized, lazy=True)
