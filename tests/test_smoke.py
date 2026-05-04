from churn.baseline import build_baseline_pipeline
from churn.data import generate_sample_dataset, split_features_target
from churn.features import build_preprocessor
from churn.model import ChurnMLP


def test_smoke_pipeline_and_model_build() -> None:
    df = generate_sample_dataset(rows=50)
    x, y = split_features_target(df.assign(Churn=df["Churn"].map({"Yes": 1, "No": 0})))
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x, y)
    model = ChurnMLP(input_dim=transformed.shape[1])
    assert transformed.shape[0] == 50
    assert model.network is not None


def test_smoke_baseline_build() -> None:
    pipeline = build_baseline_pipeline()
    assert "classifier" in pipeline.named_steps
