import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.config import MODELS_DIR, RANDOM_SEED
from churn.data import load_churn_csv, split_features_target
from churn.features import build_preprocessor
from churn.logging_config import get_logger
from churn.metrics import cost_tradeoff_metrics, find_best_threshold
from churn.train import train_mlp

logger = get_logger(__name__)

REPORTS_DIR = Path("docs") / "reports"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD = REPORTS_DIR / "model_comparison.md"


def _build_sklearn_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def compare_models(
    data_path: str | Path,
    *,
    epochs: int = 40,
    customer_value: float = 1000.0,
    retention_success_rate: float = 0.30,
    action_cost: float = 50.0,
) -> pd.DataFrame:
    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    rows = []
    mlflow.set_experiment("churn_model_comparison")
    for model_name, estimator in _build_sklearn_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        threshold, metrics = find_best_threshold(y_test, probabilities)
        cost_metrics = cost_tradeoff_metrics(
            y_test,
            probabilities,
            threshold=threshold,
            customer_value=customer_value,
            retention_success_rate=retention_success_rate,
            action_cost=action_cost,
        )
        row = {"model": model_name, "threshold": threshold} | metrics | cost_metrics
        rows.append(row)

        artifact_path = MODELS_DIR / f"{model_name}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact_path)
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "threshold_strategy": "best_f1_on_test_for_comparison",
                    "customer_value": customer_value,
                    "retention_success_rate": retention_success_rate,
                    "action_cost": action_cost,
                    "seed": RANDOM_SEED,
                }
            )
            mlflow.log_metrics({key: float(value) for key, value in row.items() if key != "model"})
            mlflow.log_artifact(str(artifact_path))

    mlp_metrics = train_mlp(data_path, epochs=epochs)
    rows.append({"model": "pytorch_mlp", **mlp_metrics})

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    COMPARISON_MD.write_text(_to_markdown_table(comparison), encoding="utf-8")
    logger.info(
        "model_comparison_created",
        csv=str(COMPARISON_CSV),
        markdown=str(COMPARISON_MD),
        models=len(comparison),
    )
    return comparison


def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in df.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--customer-value", type=float, default=1000.0)
    parser.add_argument("--retention-success-rate", type=float, default=0.30)
    parser.add_argument("--action-cost", type=float, default=50.0)
    args = parser.parse_args()
    comparison = compare_models(
        args.data,
        epochs=args.epochs,
        customer_value=args.customer_value,
        retention_success_rate=args.retention_success_rate,
        action_cost=args.action_cost,
    )
    print(comparison.to_string(index=False))
