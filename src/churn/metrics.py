"""Metricas tecnicas e de negocio usadas nos experimentos de churn."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Calcula metricas principais de classificacao binaria."""

    probabilities = np.asarray(y_prob)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def find_best_threshold(
    y_true,
    y_prob,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    # Busca um corte que privilegie F1, mais util que accuracy em churn desbalanceado.
    candidates = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.05)
    best_threshold = 0.5
    best_metrics = binary_classification_metrics(y_true, y_prob, threshold=best_threshold)

    for threshold in candidates:
        metrics = binary_classification_metrics(y_true, y_prob, threshold=float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def cost_tradeoff_metrics(
    y_true,
    y_prob,
    *,
    threshold: float,
    customer_value: float = 1000.0,
    retention_success_rate: float = 0.30,
    action_cost: float = 50.0,
) -> dict[str, float]:
    """Estima impacto financeiro simples de falsos positivos e negativos."""

    probabilities = np.asarray(y_prob)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    avoided_churn_value = tp * customer_value * retention_success_rate
    action_total_cost = (tp + fp) * action_cost
    missed_churn_cost = fn * customer_value * retention_success_rate
    return {
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
        "retention_action_cost": float(action_total_cost),
        "missed_churn_cost": float(missed_churn_cost),
        "avoided_churn_value": float(avoided_churn_value),
        "net_avoided_churn_value": float(avoided_churn_value - action_total_cost),
    }
