# Roteiro do Video - Metodo STAR

Tempo alvo: 5 minutos.

## 0:00 - 0:30 | Abertura

Apresentar o projeto:

"Este projeto resolve um problema de churn em telecom usando uma pipeline profissional de Machine
Learning. A entrega cobre EDA, baselines, rede neural MLP em PyTorch, tracking com MLflow, API
FastAPI, testes, Docker e documentacao de MLOps."

## 0:30 - 1:10 | Situation

Explicar o contexto:

"A operadora esta perdendo clientes em ritmo acelerado. Sem um score preditivo, o time de retencao
age tarde ou de forma pouco priorizada. O desafio e identificar clientes com maior risco de
cancelamento usando dados historicos de contrato, servicos e cobranca."

Mostrar rapidamente:

- `README.md`.
- Fonte do dataset no Kaggle.
- Estrutura `src/`, `docs/`, `tests/`, `notebooks/`.

## 1:10 - 1:50 | Task

Explicar o objetivo tecnico:

"A tarefa foi construir o projeto do zero ate o modelo servido via API, seguindo boas praticas:
repositorio organizado, `pyproject.toml`, reproducibilidade, validacao estratificada, baselines,
MLflow, testes, linting e Model Card."

Mostrar rapidamente:

- `docs/ml_canvas.md`.
- `notebooks/01_eda_e_baselines.ipynb`.
- `pyproject.toml`.

## 1:50 - 3:50 | Action

Demonstrar o que foi construido:

1. EDA e baselines:
   - Mostrar `notebooks/01_eda_e_baselines.ipynb`.
   - Explicar volume, qualidade, distribuicao do alvo e data readiness.
   - Mostrar DummyClassifier e Regressao Logistica registrados no MLflow.

2. Modelagem:
   - Mostrar `src/churn/train.py`.
   - Explicar MLP em PyTorch, batching, early stopping, `pos_weight` e threshold por F1.
   - Mostrar `docs/reports/model_comparison.md`.

3. Engenharia:
   - Mostrar `src/churn/api.py`.
   - Explicar `/health`, `/predict`, Pydantic, logging estruturado e middleware de latencia.
   - Rodar ou mostrar comandos:

```bash
uv run --no-editable --python 3.14.4 pytest
uv run --no-editable --python 3.14.4 ruff check .
uv run --no-editable --python 3.14.4 uvicorn --app-dir src churn.api:app --reload --reload-dir src
```

4. Deploy:
   - Mostrar `Dockerfile`.
   - Explicar escolha por API real-time.

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

## 3:50 - 4:40 | Result

Apresentar os resultados:

"A MLP ficou competitiva com os modelos Scikit-Learn, com ROC-AUC de 0.8427, PR-AUC de 0.6335,
recall de 0.7487 e F1 de 0.6215 na rodada registrada. O Random Forest teve o melhor F1, mas a MLP
foi mantida como modelo central porque era o requisito principal do projeto."

Mostrar:

- `docs/reports/model_comparison.md`.
- MLflow UI em `http://127.0.0.1:5000`, se estiver rodando.
- Swagger em `http://127.0.0.1:8000/docs`, se a API estiver rodando.

## 4:40 - 5:00 | Fechamento

Concluir:

"A entrega cobre as quatro etapas: entendimento e preparacao, modelagem neural, engenharia com API e
documentacao final. O proximo passo natural seria conectar a API a uma base real da operadora,
monitorar drift e recalibrar o threshold com custo real de campanha."

Checklist antes de gravar:

- Dataset salvo em `data/raw/churn.csv`.
- `pytest` passando.
- `ruff check .` sem erros.
- API rodando em `http://127.0.0.1:8000/docs`.
- MLflow rodando em `http://127.0.0.1:5000`.

