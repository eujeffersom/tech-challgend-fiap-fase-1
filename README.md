# Tech Challenge FIAP - Fase 1

Rede neural para previsao de churn com pipeline profissional end-to-end.

Este projeto atende ao tema central da Fase 1: criar, treinar, comparar, rastrear e servir um
modelo preditivo de churn para uma operadora de telecomunicacoes. O modelo principal e uma rede
neural MLP em PyTorch, comparada com baseline em Scikit-Learn, com experimentos rastreados no
MLflow e inferencia exposta via FastAPI.

## Tecnologias Utilizadas

- Python 3.14.4
- uv
- PyTorch
- Scikit-Learn
- MLflow
- FastAPI
- Pytest
- Ruff
- Structlog

## Estrutura do Projeto

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   `-- model_card.md
|-- models/
|-- notebooks/
|-- src/
|   `-- churn/
|       |-- api.py
|       |-- baseline.py
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- logging_config.py
|       |-- metrics.py
|       |-- model.py
|       `-- train.py
|-- tests/
|   |-- test_api.py
|   |-- test_schema.py
|   `-- test_smoke.py
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Pre-Requisitos

Instale as ferramentas abaixo antes de executar o projeto:

```bash
python --version
uv --version
git --version
```

O projeto foi configurado para Python 3.14.4:

```bash
uv python install 3.14.4
```

## Configuracao Inicial do Ambiente

Crie o ambiente virtual com a versao correta do Python:

```bash
uv venv --python 3.14.4
```

Ative o ambiente no macOS/Linux:

```bash
source .venv/bin/activate
```

Ative o ambiente no Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instale as dependencias do projeto:

```bash
uv sync --all-groups
```

## Configuracao Inicial do Git

Caso ainda nao tenha configurado seu usuario Git:

```bash
git config --global user.name "Jeffersom Goncalves"
git config --global user.email "eujeffersom@gmail.com"
```

Verifique as configuracoes:

```bash
git config --list
```

Repositorio remoto:

```bash
git remote -v
```

URL do projeto:

```text
git@github.com:eujeffersom/tech-challgend-fiap-fase-1.git
```

## Dados

O dataset principal deve ser salvo em:

```text
data/raw/churn.csv
```

A coluna alvo esperada e:

```text
Churn
```

Valores aceitos para o alvo:

```text
Yes
No
1
0
```

Caso nao tenha um dataset real, gere uma base sintetica para validar o pipeline:

```bash
uv run --python 3.14.4 churn-generate-sample --output data/raw/churn.csv --rows 1000
```

## Treinamento dos Modelos

### Baseline com Scikit-Learn

O baseline usa regressao logistica com pipeline de pre-processamento e validacao cruzada
estratificada.

```bash
uv run --python 3.14.4 churn-train-baseline --data data/raw/churn.csv
```

Artefato gerado:

```text
models/baseline_logreg.joblib
```

### Rede Neural MLP com PyTorch

O modelo principal e uma MLP treinada com PyTorch.

```bash
uv run --python 3.14.4 churn-train-mlp --data data/raw/churn.csv --epochs 40
```

Artefatos gerados:

```text
models/churn_mlp.pt
models/preprocessor.joblib
```

## MLflow

Os experimentos registram parametros, metricas e artefatos.

Metricas rastreadas:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Abra a interface do MLflow:

```bash
uv run --python 3.14.4 mlflow ui
```

Depois acesse:

```text
http://127.0.0.1:5000
```

## API de Inferencia

Inicie a API:

```bash
uv run --python 3.14.4 uvicorn churn.api:app --reload
```

Documentacao interativa:

```text
http://127.0.0.1:8000/docs
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

## Testes Automatizados

Execute todos os testes:

```bash
uv run --python 3.14.4 pytest
```

Testes implementados:

- Smoke test
- Schema test
- API test

## Linting e Formatacao

Verifique o lint:

```bash
uv run --python 3.14.4 ruff check .
```

Verifique a formatacao:

```bash
uv run --python 3.14.4 ruff format --check .
```

Aplicar formatacao automaticamente:

```bash
uv run --python 3.14.4 ruff format .
```

## Model Card

A documentacao do modelo esta em:

```text
docs/model_card.md
```

O arquivo descreve:

- Uso pretendido
- Dados de entrada
- Metricas
- Limitacoes
- Vieses e riscos
- Mitigacoes recomendadas

## DVC Opcional

Para versionar dados grandes, instale o DVC:

```bash
pip3 install dvc
```

Ou no macOS com Homebrew:

```bash
brew install dvc
```

Inicialize o DVC:

```bash
dvc init
```

Adicione um dataset:

```bash
dvc add data/raw/churn.csv
git add data/raw/churn.csv.dvc data/raw/.gitignore
git commit -m "data: track churn dataset with dvc"
```

Configure um remote local de exemplo:

```bash
dvc remote add -d localstore /tmp/dvcstore
dvc push
```

## Execucao Completa do Projeto

Fluxo recomendado do zero:

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
uv run --python 3.14.4 uvicorn churn.api:app --reload
```

## Checklist dos Requisitos

- Estrutura organizada com `src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`
- `README.md` com setup, execucao e descricao do projeto
- `pyproject.toml` como fonte unica de dependencias, linting e pytest
- `.gitignore` adequado para projetos de ML
- PyTorch para MLP
- Scikit-Learn para baseline e pre-processamento
- MLflow para tracking
- FastAPI para inferencia
- Seeds fixados
- Validacao cruzada estratificada
- Model Card
- Testes automatizados
- Logging estruturado
- Ruff sem erros

## Verificacao Final

Execute:

```bash
python --version
uv --version
uv run --python 3.14.4 pytest
uv run --python 3.14.4 ruff check .
git status
```
