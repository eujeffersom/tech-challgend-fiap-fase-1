"""Leitura, validacao e preparacao basica dos dados de churn."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandera.errors import SchemaErrors

from churn.config import ID_COLUMNS, RANDOM_SEED, TARGET_ALIASES, TARGET_COLUMN
from churn.logging_config import get_logger
from churn.schema import validate_raw_churn_schema

logger = get_logger(__name__)


def load_churn_csv(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV bruto, valida o schema e normaliza o alvo."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    try:
        validate_raw_churn_schema(df)
    except SchemaErrors as exc:
        logger.info("raw_schema_validation_failed", failures=str(exc.failure_cases.head()))
        raise
    return normalize_churn_dataframe(df)


def normalize_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza coluna alvo e converte numericos lidos como texto."""

    normalized = df.copy()
    # Aceita datasets que nomeiam o alvo como Churn ou Churn?.
    target_column = next(
        (column for column in TARGET_ALIASES if column in normalized.columns),
        None,
    )
    if target_column is None:
        raise ValueError(f"Coluna alvo ausente. Use uma destas colunas: {TARGET_ALIASES}")

    if target_column != TARGET_COLUMN:
        normalized = normalized.rename(columns={target_column: TARGET_COLUMN})

    feature_columns = infer_feature_columns(normalized)
    for column in feature_columns:
        # Converte colunas numericas lidas como texto, como TotalCharges.
        if normalized[column].dtype == "object":
            converted = pd.to_numeric(normalized[column], errors="coerce")
            if converted.notna().mean() >= 0.95:
                normalized[column] = converted

    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].astype(str).str.strip()
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].map(
        {
            "Yes": 1,
            "No": 0,
            "True": 1,
            "False": 0,
            "True.": 1,
            "False.": 0,
            "1": 1,
            "0": 0,
        }
    )
    normalized = normalized.dropna(subset=[TARGET_COLUMN])
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].astype(int)
    return normalized


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    # Remove alvo e identificadores que nao devem entrar no modelo.
    ignored = set(ID_COLUMNS + [TARGET_COLUMN])
    return [column for column in df.columns if column not in ignored]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e coluna alvo para treino."""

    return df[infer_feature_columns(df)], df[TARGET_COLUMN]


def generate_sample_dataset(rows: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Cria uma base sintetica pequena para validar o pipeline localmente."""

    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20])
    monthly = rng.normal(70, 22, rows).clip(18, 120).round(2)
    senior = rng.choice([0, 1], rows, p=[0.84, 0.16])
    internet = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[0.35, 0.45, 0.20])
    support = rng.choice(["Yes", "No", "No internet service"], rows, p=[0.30, 0.50, 0.20])
    electronic = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        rows,
        p=[0.38, 0.18, 0.22, 0.22],
    )
    churn_logit = (
        -1.3
        + 1.25 * (contract == "Month-to-month")
        + 0.85 * (internet == "Fiber optic")
        + 0.65 * (support == "No")
        + 0.55 * (electronic == "Electronic check")
        + 0.45 * senior
        - 0.035 * tenure
        + 0.012 * (monthly - 70)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_prob)

    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], rows),
            "SeniorCitizen": senior,
            "Partner": rng.choice(["Yes", "No"], rows),
            "Dependents": rng.choice(["Yes", "No"], rows, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], rows, p=[0.91, 0.09]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], rows),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], rows),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], rows),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], rows),
            "TechSupport": support,
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], rows),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], rows),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], rows, p=[0.60, 0.40]),
            "PaymentMethod": electronic,
            "MonthlyCharges": monthly,
            "TotalCharges": (monthly * tenure + rng.normal(0, 35, rows)).clip(0).round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def main_generate_sample() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/churn.csv")
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_dataset(rows=args.rows)
    df.to_csv(output, index=False)
    logger.info("sample_dataset_created", output=str(output), rows=len(df))
