import pytest
from pandera.errors import SchemaErrors

from churn.data import generate_sample_dataset, load_churn_csv, normalize_churn_dataframe
from churn.schema import validate_raw_churn_schema


def _raw_telco_frame():
    return generate_sample_dataset(rows=10)


def test_schema_requires_target_column() -> None:
    df = generate_sample_dataset(rows=10).drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Coluna alvo ausente"):
        normalize_churn_dataframe(df)


def test_schema_normalizes_churn_target() -> None:
    df = normalize_churn_dataframe(generate_sample_dataset(rows=10))
    assert set(df["Churn"].unique()).issubset({0, 1})
    assert "customerID" in df.columns


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


def test_load_churn_csv_accepts_blank_total_charges(tmp_path) -> None:
    df = _raw_telco_frame()
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = " "
    csv_path = tmp_path / "churn.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_churn_csv(csv_path)

    assert loaded["TotalCharges"].isna().sum() == 1
    assert set(loaded["Churn"].unique()).issubset({0, 1})


def test_generated_sample_can_be_loaded_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "sample_churn.csv"
    generate_sample_dataset(rows=30).to_csv(csv_path, index=False)

    loaded = load_churn_csv(csv_path)

    assert loaded.shape[0] == 30
    assert "customerID" in loaded.columns
