import pytest
from pandera.errors import SchemaErrors

from churn.data import generate_sample_dataset, normalize_churn_dataframe
from churn.schema import validate_raw_churn_schema


def _raw_telco_frame():
    df = generate_sample_dataset(rows=10)
    df.insert(0, "customerID", [f"CUST-{index}" for index in range(len(df))])
    return df


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


def test_pandera_schema_validates_raw_telco_shape() -> None:
    df = _raw_telco_frame()
    validated = validate_raw_churn_schema(df)
    assert validated.shape[0] == 10


def test_pandera_schema_rejects_invalid_churn_value() -> None:
    df = _raw_telco_frame()
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(SchemaErrors):
        validate_raw_churn_schema(df)
