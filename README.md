# Tech Challenge Fase 01: Previsao de Churn de Telecom

Este e o repositorio principal do Tech Challenge da Fase 01, focado em construir uma pipeline de
ML de ponta a ponta para previsao do cancelamento de clientes (churn), seguindo boas praticas de
engenharia de Machine Learning e MLOps.

O modelo principal da entrega e uma rede neural MLP treinada com PyTorch, comparada com baseline em
Scikit-Learn, rastreada com MLflow e servida em tempo real por uma API FastAPI.

A MLP calcula `pos_weight` automaticamente para lidar com desbalanceamento da classe churn e
seleciona um threshold de classificacao com base no melhor F1 observado na validacao cruzada.

## Arquitetura do Projeto

A arquitetura foi escolhida para permitir inferencia em tempo real atraves de uma REST API. O
pipeline contempla ingestao e validacao dos dados, pre-processamento com Scikit-Learn, treinamento
de baseline, treinamento da MLP em PyTorch, rastreamento de experimentos com MLflow e exposicao do
modelo via FastAPI.

Consulte a documentacao complementar:

- [Arquitetura de Deploy e Monitoramento](docs/deploy_architecture.md)
- [Model Card (Performance e Vieses)](docs/model_card.md)
- [ML Canvas](docs/ml_canvas.md)

## Estrutura do Repositorio

- `data/`: datasets brutos e processados. Os dados reais sao ignorados pelo Git.
- `models/`: artefatos locais, incluindo preprocessor, baseline e pesos da rede neural PyTorch.
- `notebooks/`: espaco para analise exploratoria e experimentos manuais.
- `src/`: core modular do projeto, com API, treinamento, features, modelos e metricas.
- `tests/`: bateria de testes com Pytest, incluindo smoke test, schema test e teste da API.
- `docs/`: arquivos complementares de MLOps, arquitetura, model card e ML Canvas.
- `pyproject.toml`: fonte unica de dependencias, configuracao de linting e pytest.

## Como Levantar o Projeto Localmente

### 1. Setup do Ambiente

Este projeto usa `pyproject.toml` como Single Source of Truth para as dependencias. O pre-requisito
do projeto e Python 3.14.4.

Instale a versao correta do Python:

```bash
uv python install 3.14.4
```

Crie o ambiente virtual:

```bash
uv venv --python 3.14.4
```

Para Linux / macOS:

```bash
source .venv/bin/activate
uv sync --all-groups
```

Para Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
uv sync --all-groups
```

Nota para Windows: se receber erro de execucao de scripts desabilitada ao ativar o ambiente, rode:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Verifique a instalacao:

```bash
python --version
uv --version
```

### 2. Preparar o Banco de Dados

O dataset principal deve ficar em:

```text
data/raw/churn.csv
```

A coluna alvo esperada pode ser:

```text
Churn
Churn?
```

Valores aceitos para o alvo:

```text
Yes
No
1
0
```

Para validar o pipeline sem dataset externo, gere uma base sintetica:

```bash
uv run --python 3.14.4 churn-generate-sample --output data/raw/churn.csv --rows 1000
```

### 3. Execucao do Treinamento

Treine o baseline com Scikit-Learn:

```bash
uv run --python 3.14.4 churn-train-baseline --data data/raw/churn.csv
```

Treine a rede neural MLP com PyTorch:

```bash
uv run --python 3.14.4 churn-train-mlp --data data/raw/churn.csv --epochs 40
```

Artefatos gerados localmente:

```text
models/baseline_logreg.joblib
models/churn_mlp.pt
models/preprocessor.joblib
```

### 4. Rodando a API de Inferencia

Para iniciar o servidor FastAPI:

```bash
uv run --python 3.14.4 uvicorn --app-dir src churn.api:app --reload --reload-dir src
```

Acesse o portal Swagger / Docs em:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Exemplo de predicao:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer":{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":79.5,"TotalCharges":950.0}}'
```

### 5. Checagem de Qualidade

Garantindo que o codigo nao perca formato e validando regras de negocio:

```bash
uv run --python 3.14.4 pytest
uv run --python 3.14.4 ruff check .
uv run --python 3.14.4 ruff format --check .
```

Testes implementados:

- Smoke test
- Schema validation
- API test

### 6. Fluxo Completo do Zero

```bash
uv python install 3.14.4
uv venv --python 3.14.4
source .venv/bin/activate
uv sync --all-groups
uv run --python 3.14.4 churn-generate-sample --output data/raw/churn.csv --rows 1000
uv run --python 3.14.4 churn-train-baseline --data data/raw/churn.csv
uv run --python 3.14.4 churn-train-mlp --data data/raw/churn.csv --epochs 40
uv run --python 3.14.4 pytest
uv run --python 3.14.4 ruff check .
uv run --python 3.14.4 uvicorn --app-dir src churn.api:app --reload --reload-dir src
```

## Tracking no MLflow

Para visualizar as experimentacoes e comparar o baseline Scikit-Learn com a MLP PyTorch, rode:

```bash
uv run --python 3.14.4 mlflow ui
```

Acesse:

```text
http://127.0.0.1:5000
```

Metricas rastreadas:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

## Boas Praticas Implementadas

- Seeds fixados para reprodutibilidade.
- Validacao cruzada estratificada.
- Tratamento de desbalanceamento com `pos_weight`.
- Pipelines de pre-processamento com Scikit-Learn.
- Baseline para comparacao com a MLP.
- Tracking de parametros, metricas e artefatos no MLflow.
- API FastAPI para inferencia.
- Logging estruturado com `structlog`.
- Testes automatizados com Pytest.
- Linting com Ruff.
- Model Card documentando limitacoes e vieses.

## Verificacao Final

Execute:

```bash
python --version
uv --version
uv run --python 3.14.4 pytest
uv run --python 3.14.4 ruff check .
git status
```
