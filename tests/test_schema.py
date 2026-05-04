import pytest

from churn.data import generate_sample_dataset, normalize_churn_dataframe


def test_schema_requires_target_column() -> None:
    df = generate_sample_dataset(rows=10).drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Coluna alvo ausente"):
        normalize_churn_dataframe(df)


def test_schema_normalizes_churn_target() -> None:
    df = normalize_churn_dataframe(generate_sample_dataset(rows=10))
    assert set(df["Churn"].unique()).issubset({0, 1})


def test_schema_accepts_churn_question_mark_target() -> None:
    df = generate_sample_dataset(rows=10).rename(columns={"Churn": "Churn?"})
    normalized = normalize_churn_dataframe(df)
    assert "Churn" in normalized.columns
    assert set(normalized["Churn"].unique()).issubset({0, 1})
