# Tech Challenge Fase 01: Previsao de Churn de Telecom

Pipeline de Machine Learning de ponta a ponta para prever quais clientes de uma operadora de
telecom tem risco de cancelar (churn): analise exploratoria, comparacao de modelos com
Scikit-Learn, escolha de um modelo campeao e uma API FastAPI servindo esse modelo.

O campeao atual e uma **Regressao Logistica**. Ela foi comparada com Random Forest, Gradient
Boosting e uma rede neural `MLPClassifier`, todos com a mesma validacao cruzada estratificada
de 5 folds. Uma MLP em PyTorch tambem entra na tabela como experimento extra. Os cinco modelos
empataram tecnicamente (diferencas menores que o desvio da validacao cruzada), e no empate
ficou o mais simples, mais leve e mais facil de explicar.

| Modelo | ROC-AUC (CV) | F1 (CV) | Recall (CV) | F1 (teste) |
| --- | ---: | ---: | ---: | ---: |
| **Regressao Logistica (campeao)** | 0.8459 ± 0.0124 | 0.6406 ± 0.0216 | 0.7411 ± 0.0521 | 0.6186 |
| Gradient Boosting | 0.8478 ± 0.0122 | 0.6374 ± 0.0166 | 0.7385 ± 0.0262 | 0.6257 |
| Random Forest | 0.8415 ± 0.0081 | 0.6375 ± 0.0115 | 0.7472 ± 0.0631 | 0.6274 |
| MLP PyTorch (extra) | 0.8458 ± 0.0126 | 0.6366 ± 0.0241 | 0.7739 ± 0.0321 | 0.6222 |
| MLPClassifier | 0.8410 ± 0.0138 | 0.6331 ± 0.0235 | 0.7144 ± 0.0694 | 0.6195 |

Tabela completa, com PR-AUC, threshold e impacto financeiro: `docs/reports/model_comparison.md`.

## Documentacao

- [Notebook de EDA e baselines](notebooks/01_eda_e_baselines.ipynb)
- [ML Canvas](docs/ml_canvas.md): problema, stakeholders e metricas de negocio
- [Model Card](docs/model_card.md): performance, limitacoes e vieses
- [Arquitetura de deploy](docs/deploy_architecture.md)
- [Plano de monitoramento](docs/monitoring_plan.md)

## Estrutura do repositorio

```text
src/churn/          codigo de producao
  data.py           leitura e limpeza do CSV
  schema.py         validacao do CSV bruto com Pandera
  features.py       pre-processamento (ColumnTransformer)
  metrics.py        metricas tecnicas e de custo
  baseline.py       baseline de Regressao Logistica
  compare_models.py comparacao dos modelos e escolha do campeao
  train.py          MLP PyTorch (experimento extra)
  api.py            API FastAPI
  api_models.py     formato de entrada e saida da API
tests/              testes com pytest
notebooks/          EDA e baselines
models/             champion.joblib (versionado) e demais artefatos locais
data/               dataset (fora do Git)
docs/               ML Canvas, Model Card, arquitetura, monitoramento e relatorios
```

## Pre-requisitos

- [uv](https://docs.astral.sh/uv/) 0.13 ou mais novo
- Python 3.14.4 (o `make setup` instala pelo uv)
- Docker, opcional, para rodar a API em container
- `make` (macOS e Linux ja tem; no Windows, rode os comandos `uv` mostrados no `Makefile`)

## Como rodar

### 1. Ambiente

```bash
make setup
```

Instala o Python 3.14.4 e todas as dependencias travadas no `uv.lock`. As dependencias ficam
separadas por funcao:

| Grupo | Para que serve |
| --- | --- |
| principal | o que a API precisa (FastAPI, Scikit-Learn, pandas). E so isso que vai para o Docker |
| `train` | treino e tracking (PyTorch, MLflow, Pandera, KaggleHub) |
| `notebook` | executar o notebook de EDA |
| `dev` | testes e lint |

No dia a dia os quatro sao instalados juntos, entao nao precisa se preocupar com isso.

> **macOS com a pasta no iCloud** (dentro de `Documents` ou `Desktop`): o iCloud pode tirar
> arquivos do `.venv` do disco e deixar tudo muito lento. Para evitar, crie o ambiente fora da
> sincronizacao:
>
> ```bash
> uv venv --python 3.14.4 .venv.nosync && ln -s .venv.nosync .venv && make setup
> ```

### 2. Dados

O dataset e o [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Baixe o CSV e salve como `data/raw/churn.csv`:

```bash
mkdir -p data/raw
cp ~/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/churn.csv
```

Ou baixe pelo KaggleHub:

```bash
uv run --no-editable churn-download-data --output data/raw/churn.csv
```

Para testar sem o dataset real, da para gerar uma base sintetica com `make sample`. Cuidado:
ele grava em `data/raw/churn.csv` e sobrescreve o arquivo real se ele estiver la.

### 3. Treino e comparacao

```bash
make compare-models
```

Treina os cinco modelos com a mesma validacao cruzada, escolhe o campeao (maior F1 medio na CV
entre os modelos Scikit-Learn), salva em `models/champion.joblib` e atualiza
`docs/reports/model_comparison.md`. Leva menos de um minuto.

Outros comandos:

```bash
make train-baseline   # so a Regressao Logistica de referencia
make train-mlp        # so a MLP PyTorch
make notebook         # reexecuta a EDA e salva o notebook com as saidas
make mlflow           # abre o MLflow em http://127.0.0.1:5000
```

### 4. API

O `champion.joblib` ja vem no repositorio, entao a API sobe logo depois do `make setup`, sem
precisar treinar nada:

```bash
make run
```

Swagger em http://127.0.0.1:8000/docs, com um exemplo de cliente ja preenchido.

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer":{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":79.5,"TotalCharges":950.0}}'
```

Resposta:

```json
{"churn_probability": 0.895, "churn_prediction": 1, "threshold": 0.58}
```

- `churn_probability`: probabilidade de cancelamento.
- `churn_prediction`: 1 se a probabilidade passar do threshold.
- `threshold`: corte escolhido na validacao cruzada (melhor F1), em vez do 0.5 padrao.

O payload e validado campo a campo. Tipo errado, valor negativo ou categoria que nao existe no
dataset (ex.: `"Contract": "XYZ"`) voltam **422** com a explicacao do erro. `TotalCharges` pode
vir `null` para cliente recem-chegado, como acontece no dataset original.

### 5. Qualidade

```bash
make check
```

Roda `ruff check`, `ruff format --check` e os testes. Vale rodar antes de cada commit. Os testes
cobrem:

- contrato do `/predict` (campos e tipos da resposta) e do `/health`
- respostas 422 para payload invalido e 503 sem modelo
- validacao do schema do CSV e do `TotalCharges` vazio
- comparacao de modelos: candidatos, validacao cruzada e regra do campeao
- smoke test do pipeline de pre-processamento

### 6. Docker

```bash
make docker-build
make docker-run
```

A imagem tem dois estagios e leva so o necessario para servir o modelo: Python, as dependencias
principais e o `champion.joblib`. PyTorch, MLflow, notebooks, testes e dados ficam de fora.
Roda com usuario sem privilegios e tem healthcheck em `/health`.

## Boas praticas aplicadas

- Seed fixa (`RANDOM_SEED = 42`) em split, validacao cruzada e modelos.
- Mesma validacao cruzada estratificada para todos os modelos, com threshold escolhido so com
  dados de treino. O conjunto de teste e usado uma unica vez, no fim.
- Pre-processamento dentro do `Pipeline`, ajustado so no treino de cada fold.
- Separacao entre experimentacao (`notebooks/`) e codigo de producao (`src/churn/`).
- Dependencias travadas no `uv.lock` e separadas por funcao.
- Validacao do CSV bruto com Pandera e do payload da API com Pydantic.
- Logs estruturados em JSON (`structlog`) e header `X-Process-Time-ms` com a latencia.
- Experimentos registrados no MLflow, com hash do dataset.
- Lint e formatacao com Ruff, testes com Pytest.
- Model Card com limitacoes e vieses medidos por segmento.
