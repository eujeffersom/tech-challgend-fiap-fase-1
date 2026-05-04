import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from churn.config import MLP_MODEL_PATH, PREPROCESSOR_PATH, RANDOM_SEED
from churn.data import load_churn_csv, split_features_target
from churn.features import build_preprocessor
from churn.logging_config import get_logger
from churn.metrics import binary_classification_metrics, find_best_threshold
from churn.model import ChurnMLP, set_global_seed

logger = get_logger(__name__)


def _to_tensor(features: np.ndarray, target: np.ndarray | None = None):
    x_tensor = torch.tensor(features, dtype=torch.float32)
    if target is None:
        return x_tensor
    y_tensor = torch.tensor(target, dtype=torch.float32)
    return x_tensor, y_tensor


def _train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    pos_weight: float,
    threshold: float | None = None,
) -> tuple[ChurnMLP, dict[str, float]]:
    model = ChurnMLP(input_dim=x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    train_x, train_y = _to_tensor(x_train, y_train)
    valid_x = _to_tensor(x_valid)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_prob = torch.sigmoid(model(valid_x)).numpy()
    if threshold is None:
        selected_threshold, metrics = find_best_threshold(y_valid, valid_prob)
        metrics["threshold"] = selected_threshold
    else:
        metrics = binary_classification_metrics(y_valid, valid_prob, threshold=threshold)
        metrics["threshold"] = threshold
    return model, metrics


def _calculate_pos_weight(y_train: np.ndarray) -> float:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def train_mlp(
    data_path: str | Path,
    *,
    epochs: int = 40,
    lr: float = 0.001,
    hidden_dim: int = 64,
    dropout: float = 0.2,
) -> dict[str, float]:
    set_global_seed(RANDOM_SEED)
    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()
    pos_weight = _calculate_pos_weight(y_train_array)

    params = {
        "model_type": "pytorch_mlp",
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "cv_folds": 5,
        "seed": RANDOM_SEED,
        "pos_weight": pos_weight,
    }

    with mlflow.start_run(run_name="pytorch_mlp"):
        mlflow.log_params(params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_metrics: list[dict[str, float]] = []
        cv_splits = cv.split(x_train_transformed, y_train_array)
        for fold, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
            _, metrics = _train_single_model(
                x_train_transformed[train_idx],
                y_train_array[train_idx],
                x_train_transformed[valid_idx],
                y_train_array[valid_idx],
                epochs=epochs,
                lr=lr,
                hidden_dim=hidden_dim,
                dropout=dropout,
                pos_weight=_calculate_pos_weight(y_train_array[train_idx]),
            )
            fold_metrics.append(metrics)
            mlflow.log_metrics({f"cv_fold_{fold}_{key}": value for key, value in metrics.items()})

        selected_threshold = float(
            np.mean([fold_result["threshold"] for fold_result in fold_metrics])
        )
        mlflow.log_metric("selected_threshold", selected_threshold)

        for metric_name in fold_metrics[0]:
            mlflow.log_metric(
                f"cv_{metric_name}",
                float(np.mean([fold_result[metric_name] for fold_result in fold_metrics])),
            )

        model, test_metrics = _train_single_model(
            x_train_transformed,
            y_train_array,
            x_test_transformed,
            y_test_array,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pos_weight=pos_weight,
            threshold=selected_threshold,
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": x_train_transformed.shape[1],
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "threshold": selected_threshold,
            },
            MLP_MODEL_PATH,
        )
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        mlflow.log_artifact(str(MLP_MODEL_PATH))
        mlflow.log_artifact(str(PREPROCESSOR_PATH))

    logger.info("mlp_trained", model_path=str(MLP_MODEL_PATH), **test_metrics)
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    train_mlp(
        args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
