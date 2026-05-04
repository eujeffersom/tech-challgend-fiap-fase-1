import pytest

from churn.data import generate_sample_dataset, normalize_churn_dataframe


def test_schema_requires_target_column() -> None:
    df = generate_sample_dataset(rows=10).drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Colunas obrigatorias ausentes"):
        normalize_churn_dataframe(df)


def test_schema_normalizes_churn_target() -> None:
    df = normalize_churn_dataframe(generate_sample_dataset(rows=10))
    assert set(df["Churn"].unique()).issubset({0, 1})
