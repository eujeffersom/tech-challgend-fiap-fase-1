"""Treinamento da MLP PyTorch com CV estratificada, early stopping e MLflow."""

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from churn.config import MLP_MODEL_PATH, PREPROCESSOR_PATH, RANDOM_SEED
from churn.data import load_churn_csv, split_features_target
from churn.features import build_preprocessor
from churn.logging_config import get_logger
from churn.metrics import binary_classification_metrics, cost_tradeoff_metrics, find_best_threshold
from churn.model import ChurnMLP, set_global_seed

logger = get_logger(__name__)


def _to_tensor(features: np.ndarray, target: np.ndarray | None = None):
    """Converte arrays do preprocessor para tensores PyTorch."""

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
    batch_size: int,
    patience: int,
    threshold: float | None = None,
) -> tuple[ChurnMLP, dict[str, float]]:
    """Treina uma MLP em um split e retorna metricas de validacao."""

    model = ChurnMLP(input_dim=x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # pos_weight penaliza mais erros na classe churn, que costuma ser minoritaria.
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    train_x, train_y = _to_tensor(x_train, y_train)
    valid_x = _to_tensor(x_valid)
    valid_y = torch.tensor(y_valid, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y).item())
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "early_stopping_triggered",
                epoch=epoch + 1,
                best_valid_loss=best_valid_loss,
            )
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        valid_prob = torch.sigmoid(model(valid_x)).numpy()
    if threshold is None:
        # Nos folds de validacao, escolhemos o melhor threshold por F1.
        selected_threshold, metrics = find_best_threshold(y_valid, valid_prob)
        metrics["threshold"] = selected_threshold
    else:
        metrics = binary_classification_metrics(y_valid, valid_prob, threshold=threshold)
        metrics["threshold"] = threshold
    metrics["best_valid_loss"] = best_valid_loss
    return model, metrics


def _calculate_pos_weight(y_train: np.ndarray) -> float:
    # Razao negativos/positivos usada pelo BCEWithLogitsLoss.
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
    batch_size: int = 128,
    patience: int = 8,
) -> dict[str, float]:
    """Treina a MLP final, registra no MLflow e salva artefatos."""

    set_global_seed(RANDOM_SEED)
    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    # Separacao final: 80% treino e 20% teste, preservando a proporcao do alvo.
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
        "batch_size": batch_size,
        "patience": patience,
        "cv_folds": 5,
        "seed": RANDOM_SEED,
        "pos_weight": pos_weight,
    }

    with mlflow.start_run(run_name="pytorch_mlp"):
        mlflow.log_params(params)
        # A validacao cruzada estima metricas e thresholds usando apenas o treino.
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
                batch_size=batch_size,
                patience=patience,
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
            batch_size=batch_size,
            patience=patience,
            threshold=selected_threshold,
        )
        with torch.no_grad():
            test_prob = torch.sigmoid(model(_to_tensor(x_test_transformed))).numpy()
        test_cost_metrics = cost_tradeoff_metrics(
            y_test_array,
            test_prob,
            threshold=selected_threshold,
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics({f"cost_{key}": value for key, value in test_cost_metrics.items()})

        MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": x_train_transformed.shape[1],
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "threshold": selected_threshold,
                "batch_size": batch_size,
                "patience": patience,
            },
            MLP_MODEL_PATH,
        )
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        mlflow.log_artifact(str(MLP_MODEL_PATH))
        mlflow.log_artifact(str(PREPROCESSOR_PATH))

    output_metrics = test_metrics | test_cost_metrics
    logger.info("mlp_trained", model_path=str(MLP_MODEL_PATH), **output_metrics)
    return output_metrics


def main() -> None:
    """Entrada de linha de comando para treinar a MLP."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    train_mlp(
        args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        patience=args.patience,
    )
