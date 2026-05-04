PYTHON_VERSION := 3.14.4
DATA_PATH := data/raw/churn.csv
IMAGE_NAME := telco-churn-api

.PHONY: setup sample train-baseline train-mlp compare-models test lint format run mlflow docker-build docker-run

setup:
	uv python install $(PYTHON_VERSION)
	uv venv --python $(PYTHON_VERSION)
	uv sync --all-groups --no-editable

sample:
	uv run --no-editable --python $(PYTHON_VERSION) churn-generate-sample --output $(DATA_PATH) --rows 1000

train-baseline:
	uv run --no-editable --python $(PYTHON_VERSION) churn-train-baseline --data $(DATA_PATH)

train-mlp:
	uv run --no-editable --python $(PYTHON_VERSION) churn-train-mlp --data $(DATA_PATH) --epochs 40

compare-models:
	uv run --no-editable --python $(PYTHON_VERSION) churn-compare-models --data $(DATA_PATH) --epochs 40

test:
	uv run --no-editable --python $(PYTHON_VERSION) pytest

lint:
	uv run --no-editable --python $(PYTHON_VERSION) ruff check .

format:
	uv run --no-editable --python $(PYTHON_VERSION) ruff format .

run:
	uv run --no-editable --python $(PYTHON_VERSION) uvicorn --app-dir src churn.api:app --reload --reload-dir src

mlflow:
	uv run --no-editable --python $(PYTHON_VERSION) mlflow ui

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p 8000:8000 $(IMAGE_NAME)
