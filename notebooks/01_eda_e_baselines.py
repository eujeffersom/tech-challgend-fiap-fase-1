"""EDA e comparacao de baselines para o dataset de churn.

Execute a partir da raiz do projeto:

    uv run --python 3.14.4 python notebooks/01_eda_e_baselines.py
"""

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # Permite executar o script sem instalar o pacote em modo notebook.
    sys.path.insert(0, str(SRC_DIR))

from churn.config import RANDOM_SEED  # noqa: E402
from churn.data import load_churn_csv, split_features_target  # noqa: E402
from churn.features import build_preprocessor  # noqa: E402
from churn.metrics import binary_classification_metrics  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
DATASET_SOURCE = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
DATASET_ORIGINAL_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def calculate_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_dataset(df: pd.DataFrame) -> None:
    print("Dimensoes:", df.shape)
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nTipos de dados:")
    print(df.dtypes)
    print("\nValores ausentes por coluna:")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\nDistribuicao do alvo:")
    print(df["Churn"].value_counts(normalize=True).rename("proportion"))


def data_readiness_report(df: pd.DataFrame) -> dict[str, float | int]:
    duplicated_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    missing_rate = float(missing_cells / df.size)
    target_rate = float(df["Churn"].mean())
    high_cardinality_columns = [
        column
        for column in df.select_dtypes(include=["object"]).columns
        if df[column].nunique() > 50
    ]

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicated_rows": duplicated_rows,
        "missing_cells": missing_cells,
        "missing_rate": missing_rate,
        "target_churn_rate": target_rate,
        "high_cardinality_columns": len(high_cardinality_columns),
    }

    print("\nData readiness:")
    for key, value in report.items():
        print(f"- {key}: {value}")
    print("- readiness_status:", "ready" if missing_rate < 0.05 and target_rate > 0 else "review")
    return report


def plot_target_distribution(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Nao churn", "Churn"]
    ax = counts.plot(kind="bar", color=["#4C78A8", "#F58518"])
    ax.set_title("Distribuicao da Classe Churn")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")
    ax.set_xticklabels(labels, rotation=0)
    plt.tight_layout()
    figure_path = FIGURES_DIR / "target_distribution.png"
    plt.savefig(figure_path, dpi=120)
    plt.close()
    print(f"\nGrafico salvo em: {figure_path}")


def evaluate_baselines(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # Usa o mesmo split 80/20 e validacao estratificada do pipeline principal.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }

    rows = []
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("classifier", estimator),
            ]
        )
        cv_results = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring)
        pipeline.fit(x_train, y_train)

        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(x_test)[:, 1]
        else:
            y_prob = pipeline.predict(x_test)

        test_metrics = binary_classification_metrics(y_test, y_prob)
        test_metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
        row = {"model": model_name}
        row.update(
            {f"cv_{metric}": float(np.mean(cv_results[f"test_{metric}"])) for metric in scoring}
        )
        row.update({f"test_{metric}": value for metric, value in test_metrics.items()})
        rows.append(row)

    return pd.DataFrame(rows).sort_values("test_f1", ascending=False)


def log_baselines_to_mlflow(results: pd.DataFrame, readiness: dict[str, float | int]) -> None:
    dataset_hash = calculate_file_hash(DATA_PATH)
    mlflow.set_experiment("churn_eda_baselines")

    for row in results.to_dict(orient="records"):
        model_name = str(row["model"])
        with mlflow.start_run(run_name=f"eda_{model_name}"):
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "dataset_source": DATASET_SOURCE,
                    "dataset_original_file": DATASET_ORIGINAL_FILE,
                    "dataset_local_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
                    "dataset_sha256": dataset_hash,
                    "cv_folds": 5,
                    "split": "80_20_stratified",
                    "seed": RANDOM_SEED,
                }
            )
            mlflow.log_metrics({key: float(value) for key, value in row.items() if key != "model"})
            mlflow.log_metrics({f"data_{key}": float(value) for key, value in readiness.items()})


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em {DATA_PATH}. "
            "Coloque o churn.csv em data/raw/ ou gere uma base sintetica."
        )

    df = load_churn_csv(DATA_PATH)
    describe_dataset(df)
    readiness = data_readiness_report(df)
    plot_target_distribution(df)

    x, y = split_features_target(df)
    results = evaluate_baselines(x, y)
    log_baselines_to_mlflow(results, readiness)
    print("\nResultados dos baselines:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
