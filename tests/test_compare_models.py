import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn.compare_models import (
    CV_METRICS,
    build_candidate_models,
    cross_validate_with_threshold,
    select_champion,
)
from churn.data import generate_sample_dataset, normalize_churn_dataframe, split_features_target
from churn.features import build_preprocessor


def _sample_split():
    df = normalize_churn_dataframe(generate_sample_dataset(rows=200))
    return split_features_target(df)


def test_candidates_include_linear_tree_and_mlp() -> None:
    models = build_candidate_models()
    assert {"logistic_regression", "random_forest", "mlp_classifier"} <= set(models)
    assert isinstance(models["mlp_classifier"], MLPClassifier)


def test_cross_validation_returns_threshold_and_summary() -> None:
    x, y = _sample_split()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x)),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    threshold, summary = cross_validate_with_threshold(pipeline, x, y, n_splits=3)

    assert 0.0 < threshold < 1.0
    for metric in CV_METRICS:
        assert 0.0 <= summary[f"cv_{metric}_mean"] <= 1.0
        assert summary[f"cv_{metric}_std"] >= 0.0


def test_champion_is_best_cv_f1_among_candidates() -> None:
    comparison = pd.DataFrame(
        {
            "model": ["pytorch_mlp", "random_forest", "logistic_regression"],
            "cv_f1_mean": [0.70, 0.62, 0.64],
        }
    )
    # pytorch_mlp tem o maior F1, mas nao e candidato a campeao.
    champion = select_champion(comparison, candidates=["random_forest", "logistic_regression"])
    assert champion == "logistic_regression"
