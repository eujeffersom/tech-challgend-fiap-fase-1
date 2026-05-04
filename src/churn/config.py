import os
from pathlib import Path

RANDOM_SEED = 42
TARGET_COLUMN = "Churn"
TARGET_ALIASES = ["Churn", "Churn?"]
ID_COLUMNS = ["customerID", "Phone"]

PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", Path.cwd())).resolve()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MLP_MODEL_PATH = MODELS_DIR / "churn_mlp.pt"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logreg.joblib"

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
