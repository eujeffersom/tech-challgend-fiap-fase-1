import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Define pre-processamento dinamico para aceitar datasets Telco e sinteticos.
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    return categorical_columns, numeric_columns


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_columns, numeric_columns = infer_column_types(df)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def validate_feature_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Payload de entrada vazio.")
