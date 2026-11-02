"""Compara os modelos candidatos usando a mesma validacao cruzada para todos."""

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn.config import MODELS_DIR, RANDOM_SEED
from churn.data import load_churn_csv, split_features_target
from churn.features import build_preprocessor
from churn.logging_config import get_logger
from churn.metrics import binary_classification_metrics, cost_tradeoff_metrics, find_best_threshold
from churn.train import train_mlp

logger = get_logger(__name__)

REPORTS_DIR = Path("docs") / "reports"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD = REPORTS_DIR / "model_comparison.md"

CV_FOLDS = 5
# Metricas que entram na tabela com media e desvio dos folds.
CV_METRICS = ["roc_auc", "pr_auc", "f1", "recall", "precision"]


def build_candidate_models() -> dict[str, object]:
    """Modelos do Scikit-Learn que entram na disputa."""

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
        # O MLPClassifier nao tem class_weight. O desbalanceamento fica por conta
        # do threshold, que e escolhido na validacao cruzada.
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=RANDOM_SEED,
        ),
    }


def cross_validate_with_threshold(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = CV_FOLDS,
) -> tuple[float, dict[str, float]]:
    """Roda a CV e escolhe o threshold sem encostar no conjunto de teste.

    Em cada fold o modelo treina, preve o fold de validacao e acha o corte com
    melhor F1. O threshold final e a media desses cortes, igual ao que a MLP
    PyTorch faz em train.py. Assim todos os modelos sao avaliados do mesmo jeito.
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    for train_idx, valid_idx in cv.split(x_train, y_train):
        fold_model = clone(pipeline)
        fold_model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
        valid_prob = fold_model.predict_proba(x_train.iloc[valid_idx])[:, 1]
        threshold, metrics = find_best_threshold(y_train.iloc[valid_idx], valid_prob)
        fold_results.append(metrics | {"threshold": threshold})

    summary = {}
    for metric in CV_METRICS:
        values = [fold[metric] for fold in fold_results]
        summary[f"cv_{metric}_mean"] = float(np.mean(values))
        summary[f"cv_{metric}_std"] = float(np.std(values))

    threshold = float(np.mean([fold["threshold"] for fold in fold_results]))
    return threshold, summary


def evaluate_on_test(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float,
    customer_value: float,
    retention_success_rate: float,
    action_cost: float,
) -> dict[str, float]:
    """Mede o modelo ja treinado no teste, usando o threshold que veio da CV."""

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = binary_classification_metrics(y_test, probabilities, threshold=threshold)
    costs = cost_tradeoff_metrics(
        y_test,
        probabilities,
        threshold=threshold,
        customer_value=customer_value,
        retention_success_rate=retention_success_rate,
        action_cost=action_cost,
    )
    return {f"test_{key}": value for key, value in metrics.items()} | costs


def compare_models(
    data_path: str | Path,
    *,
    epochs: int = 40,
    customer_value: float = 1000.0,
    retention_success_rate: float = 0.30,
    action_cost: float = 50.0,
    include_pytorch: bool = True,
) -> pd.DataFrame:
    """Treina todos os candidatos e grava a tabela comparativa em CSV e Markdown."""

    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    # Mesmo split 80/20 do baseline e da MLP PyTorch.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    cost_params = {
        "customer_value": customer_value,
        "retention_success_rate": retention_success_rate,
        "action_cost": action_cost,
    }

    rows = []
    mlflow.set_experiment("churn_model_comparison")
    for model_name, estimator in build_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("classifier", estimator),
            ]
        )
        threshold, cv_summary = cross_validate_with_threshold(pipeline, x_train, y_train)

        # Depois da CV, treina com todo o treino e mede uma unica vez no teste.
        pipeline.fit(x_train, y_train)
        test_summary = evaluate_on_test(
            pipeline, x_test, y_test, threshold=threshold, **cost_params
        )
        row = {"model": model_name, "threshold": threshold} | cv_summary | test_summary
        rows.append(row)
        logger.info("model_evaluated", model=model_name, cv_f1=cv_summary["cv_f1_mean"])

        artifact_path = MODELS_DIR / f"{model_name}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact_path)
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "cv_folds": CV_FOLDS,
                    "threshold_strategy": "media_dos_folds_melhor_f1",
                    "seed": RANDOM_SEED,
                }
                | cost_params
            )
            mlflow.log_metrics({key: float(value) for key, value in row.items() if key != "model"})
            mlflow.log_artifact(str(artifact_path))

    if include_pytorch:
        # A MLP PyTorch segue como experimento extra. Ela ja faz a mesma CV
        # internamente, so renomeamos as chaves para casar com a tabela.
        mlp = train_mlp(data_path, epochs=epochs)
        rows.append(_pytorch_row(mlp))

    comparison = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
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


def _pytorch_row(mlp_metrics: dict[str, float]) -> dict[str, float | str]:
    test_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    row: dict[str, float | str] = {"model": "pytorch_mlp", "threshold": mlp_metrics["threshold"]}
    row |= {key: value for key, value in mlp_metrics.items() if key.startswith("cv_")}
    row |= {f"test_{key}": mlp_metrics[key] for key in test_keys}
    row |= {
        key: value
        for key, value in mlp_metrics.items()
        if key not in test_keys and not key.startswith("cv_") and key != "threshold"
    }
    row.pop("best_valid_loss", None)
    return row


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Tabela enxuta para o README e o Model Card. O CSV guarda todas as colunas."""

    columns = {
        "model": "Modelo",
        "cv_roc_auc_mean": "ROC-AUC (CV)",
        "cv_pr_auc_mean": "PR-AUC (CV)",
        "cv_f1_mean": "F1 (CV)",
        "cv_recall_mean": "Recall (CV)",
        "threshold": "Threshold",
        "test_f1": "F1 (teste)",
        "test_roc_auc": "ROC-AUC (teste)",
        "net_avoided_churn_value": "Valor liquido (teste)",
    }
    header = "| " + " | ".join(columns.values()) + " |"
    separator = "| --- |" + " ---: |" * (len(columns) - 1)
    lines = [header, separator]
    for record in df.to_dict(orient="records"):
        cells = [str(record["model"])]
        for key in list(columns)[1:]:
            value = float(record[key])
            std_key = key.replace("_mean", "_std")
            if key.endswith("_mean") and std_key in record:
                cells.append(f"{value:.4f} ± {float(record[std_key]):.4f}")
            elif key == "net_avoided_churn_value":
                cells.append(f"{value:,.0f}")
            elif key == "threshold":
                cells.append(f"{value:.2f}")
            else:
                cells.append(f"{value:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Linha de comando para gerar a comparacao completa."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--customer-value", type=float, default=1000.0)
    parser.add_argument("--retention-success-rate", type=float, default=0.30)
    parser.add_argument("--action-cost", type=float, default=50.0)
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Pula a MLP PyTorch para a comparacao rodar mais rapido.",
    )
    args = parser.parse_args()
    comparison = compare_models(
        args.data,
        epochs=args.epochs,
        customer_value=args.customer_value,
        retention_success_rate=args.retention_success_rate,
        action_cost=args.action_cost,
        include_pytorch=not args.skip_pytorch,
    )
    logger.info("comparison_finished", rows=comparison.to_dict(orient="records"))
