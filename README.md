# Tech Challenge FIAP - Previsao de Churn com MLP

Projeto end-to-end para previsao de churn em telecom usando rede neural MLP com PyTorch,
baselines com Scikit-Learn, rastreamento de experimentos com MLflow e API de inferencia com
FastAPI.

## Requisitos

- Python 3.14.4
- uv

> Observacao: o projeto declara `requires-python = ">=3.14.4,<3.15"`. Caso alguma wheel ainda
> nao esteja disponivel para o seu sistema operacional, o `uv sync` vai reportar o pacote
> especifico durante a resolucao.

## Setup

```bash
uv python install 3.14.4
uv venv --python 3.14.4
source .venv/bin/activate
uv sync --all-groups
```

No Windows PowerShell:

```powershell
uv python install 3.14.4
uv venv --python 3.14.4
.venv\Scripts\Activate.ps1
uv sync --all-groups
```

## Dados

Coloque um CSV em `data/raw/churn.csv` com a coluna alvo `Churn`.

Para testar o pipeline sem dataset externo:

```bash
uv run --python 3.14.4 churn-generate-sample --output data/raw/churn.csv
```

## Treino

Baseline Scikit-Learn com validacao cruzada estratificada:

```bash
uv run --python 3.14.4 churn-train-baseline --data data/raw/churn.csv
```

MLP PyTorch com MLflow:

```bash
uv run --python 3.14.4 churn-train-mlp --data data/raw/churn.csv --epochs 40
```

Visualizar experimentos:

```bash
uv run --python 3.14.4 mlflow ui
```

## API

Depois do treino, inicie a API:

```bash
uv run --python 3.14.4 uvicorn churn.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predicao:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer":{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":79.5,"TotalCharges":950.0}}'
```

## Qualidade

```bash
uv run --python 3.14.4 ruff check .
uv run --python 3.14.4 ruff format --check .
uv run --python 3.14.4 pytest
```

## Estrutura

```text
src/churn/       Codigo de aplicacao, treino, features e API
data/            Dados brutos e processados
models/          Artefatos treinados
tests/           Testes automatizados
notebooks/       Analises exploratorias
docs/            Documentacao, incluindo Model Card
```
