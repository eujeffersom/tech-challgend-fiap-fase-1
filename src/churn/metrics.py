import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    probabilities = np.asarray(y_prob)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def find_best_threshold(
    y_true,
    y_prob,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.05)
    best_threshold = 0.5
    best_metrics = binary_classification_metrics(y_true, y_prob, threshold=best_threshold)

    for threshold in candidates:
        metrics = binary_classification_metrics(y_true, y_prob, threshold=float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
