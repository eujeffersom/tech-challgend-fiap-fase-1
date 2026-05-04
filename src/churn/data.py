import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, RANDOM_SEED, TARGET_COLUMN
from churn.logging_config import get_logger

logger = get_logger(__name__)


def load_churn_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    return normalize_churn_dataframe(df)


def normalize_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    missing = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]) - set(normalized.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing)}")

    normalized["TotalCharges"] = pd.to_numeric(normalized["TotalCharges"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=NUMERIC_COLUMNS + [TARGET_COLUMN])
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    normalized = normalized.dropna(subset=[TARGET_COLUMN])
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].astype(int)
    return normalized


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS], df[TARGET_COLUMN]


def generate_sample_dataset(rows: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
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
