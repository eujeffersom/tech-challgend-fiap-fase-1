import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from churn.config import BASELINE_MODEL_PATH, RANDOM_SEED
from churn.data import load_churn_csv, split_features_target
from churn.features import build_preprocessor
from churn.logging_config import get_logger
from churn.metrics import binary_classification_metrics
from churn.model import set_global_seed

logger = get_logger(__name__)


def build_baseline_pipeline() -> Pipeline:
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline(
        steps=[
            ("preprocessor", "passthrough"),
            ("classifier", classifier),
        ]
    )


def train_baseline(data_path: str | Path) -> dict[str, float]:
    set_global_seed(RANDOM_SEED)
    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    pipeline = build_baseline_pipeline()
    pipeline.set_params(preprocessor=build_preprocessor(x_train))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    with mlflow.start_run(run_name="baseline_logistic_regression"):
        cv_results = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring)
        for metric_name in scoring:
            value = float(np.mean(cv_results[f"test_{metric_name}"]))
            mlflow.log_metric(f"cv_{metric_name}", value)

        pipeline.fit(x_train, y_train)
        y_prob = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = binary_classification_metrics(y_test, y_prob)
        mlflow.log_params({"model_type": "logistic_regression", "cv_folds": 5})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, BASELINE_MODEL_PATH)
        mlflow.log_artifact(str(BASELINE_MODEL_PATH))

    logger.info("baseline_trained", model_path=str(BASELINE_MODEL_PATH), **test_metrics)
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    args = parser.parse_args()
    train_baseline(args.data)
