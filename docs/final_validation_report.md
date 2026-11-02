# Relatorio de Validacao Final

Data da validacao: 2026-05-04

## Resumo Executivo

O projeto esta bem estruturado e atende grande parte dos requisitos de engenharia, documentacao,
API, testes e rastreamento. A API de inferencia respondeu corretamente usando os artefatos locais
atuais em `models/`.

Porem, a validacao final encontrou dois pontos que impedem considerar o projeto 100% pronto:

1. Os comandos de treino falham com o `data/raw/churn.csv` real do Kaggle por causa da coluna
   `TotalCharges`, que possui valores em branco no dataset original.
2. A base sintetica gerada pelo comando `churn-generate-sample` nao passa no schema Pandera quando
   usada novamente pelos comandos de treino, porque nao contem a coluna `customerID`.

Conclusao: o projeto esta proximo de compliance total, mas precisa corrigir a etapa de carregamento
e validacao de dados antes da entrega final.

## Comandos Executados

### Qualidade de codigo

```bash
uv run --no-editable --python 3.14.4 ruff check .
```

Resultado: aprovado.

```text
All checks passed!
```

```bash
uv run --no-editable --python 3.14.4 pytest
```

Resultado: aprovado.

```text
9 passed
```

### Formatacao

```bash
uv run --no-editable --python 3.14.4 ruff format --check .
```

Resultado: reprovado para dois arquivos.

```text
Would reformat: notebooks/01_eda_e_baselines.ipynb
Would reformat: notebooks/01_eda_e_baselines.py
2 files would be reformatted, 16 files already formatted
```

### API de inferencia

Foi executada uma chamada real ao endpoint `/predict` em memoria, usando os artefatos atuais em
`models/`.

Resultado: aprovado.

```text
status_code: 200
response: {"churn_probability":0.8009823560714722,"churn_prediction":1,"threshold":0.53}
```

### Geracao de base sintetica

```bash
uv run --no-editable --python 3.14.4 churn-generate-sample --output /private/tmp/churn_validation.csv --rows 300
```

Resultado: aprovado. O CSV temporario foi gerado.

### Treino baseline com base sintetica

```bash
CHURN_PROJECT_ROOT=/private/tmp/churn_validation_root \
MLFLOW_TRACKING_URI=file:///private/tmp/churn_validation_mlruns \
uv run --no-editable --python 3.14.4 churn-train-baseline --data /private/tmp/churn_validation.csv
```

Resultado: reprovado.

Motivo principal:

```text
column 'customerID' not in dataframe
```

### Treino baseline com dataset real

```bash
CHURN_PROJECT_ROOT=/private/tmp/churn_validation_root \
MLFLOW_TRACKING_URI=file:///private/tmp/churn_validation_mlruns \
uv run --no-editable --python 3.14.4 churn-train-baseline --data data/raw/churn.csv
```

Resultado: reprovado.

Motivo principal:

```text
Error while coercing 'TotalCharges' to type float64
expected series 'TotalCharges' to have type float64, got object
```

## Achados por Severidade

### Critico - Treino falha com dataset real

Arquivo relacionado: `src/churn/data.py`

Trecho relevante:

- `load_churn_csv` le o CSV.
- Em seguida chama `validate_raw_churn_schema(df)`.
- So depois chamaria `normalize_churn_dataframe(df)`.

Problema: o dataset original do Kaggle possui valores em branco em `TotalCharges`. Como o schema
Pandera tenta validar/coagir `TotalCharges` para `float` antes da limpeza, a validacao falha.

Impacto:

- `churn-train-baseline --data data/raw/churn.csv` falha.
- `churn-train-mlp --data data/raw/churn.csv` tende a falhar pelo mesmo motivo.
- `churn-compare-models --data data/raw/churn.csv` tende a falhar pelo mesmo motivo.
- O notebook tambem usa `load_churn_csv`, entao pode falhar ao ser reexecutado com o CSV real.

Recomendacao:

- Limpar `TotalCharges` antes da validacao Pandera, convertendo espacos em branco para `NaN`.
- Ou ajustar o schema para aceitar a coluna bruta como string e validar a versao normalizada depois.

### Alto - Base sintetica nao passa no proprio comando de treino

Arquivo relacionado: `src/churn/data.py`

Problema: `generate_sample_dataset` nao cria a coluna `customerID`, mas `TELCO_CHURN_SCHEMA` exige
`customerID`.

Impacto:

- O comando documentado para gerar uma base sintetica funciona.
- Mas o CSV gerado por ele falha quando usado em `churn-train-baseline`.

Recomendacao:

- Adicionar `customerID` na base sintetica.
- Criar teste que gere o CSV sintetico, carregue com `load_churn_csv` e execute pelo menos uma etapa
  simples do pipeline.

### Medio - Checagem de formatacao falha em notebooks

Arquivos:

- `notebooks/01_eda_e_baselines.ipynb`
- `notebooks/01_eda_e_baselines.py`

Impacto:

- `ruff check .` passa.
- `ruff format --check .` falha.

Recomendacao:

- Executar `uv run --no-editable --python 3.14.4 ruff format notebooks/01_eda_e_baselines.py`.
- Para o `.ipynb`, avaliar se o time quer formatar notebooks pelo Ruff ou excluir notebooks da
  checagem de formatacao.

### Medio - Testes nao cobrem falha end-to-end de dados

Arquivos:

- `tests/test_schema.py`
- `tests/test_smoke.py`
- `tests/test_api.py`

Observacoes:

- O teste de schema insere `customerID` manualmente antes de validar.
- O smoke test usa `generate_sample_dataset`, mas nao passa pelo caminho `load_churn_csv`.
- O teste de `/predict` aceita `200`, `422` ou `503`, entao ele nao garante que a inferencia esta
  operacional quando artefatos existem.

Recomendacao:

- Adicionar teste end-to-end com CSV sintetico salvo em disco.
- Adicionar teste para `TotalCharges` em branco, como existe no dataset Kaggle.
- Adicionar teste de `/predict` que exija `200` quando artefatos temporarios validos estiverem
  presentes.

## Validacao de Documentacao

Artefatos documentais encontrados:

- `README.md`: contem setup, dataset, treino, API, Docker, MLflow e verificacao final.
- `docs/ml_canvas.md`: contem stakeholders, metricas de negocio, SLOs e riscos.
- `docs/model_card.md`: contem performance, limitacoes, vieses e cenarios de falha.
- `docs/deploy_architecture.md`: documenta escolha por API real-time, comparacao com batch e deploy.
- `docs/monitoring_plan.md`: contem metricas, alertas e playbook de resposta.
- `docs/reports/model_comparison.md`: contem tabela comparativa de modelos.
- `models/README.md`: contem artefatos locais esperados no diretorio `models/`.
- `data/README.md`: contem fonte do dataset e caminho esperado.

Status: documentacao principal esta adequada. O principal risco e que os comandos documentados de
treino precisam voltar a funcionar com o CSV real.

## Validacao de Requisitos

| Requisito | Status | Observacao |
| --- | --- | --- |
| Estrutura organizada | Aprovado | Diretorios principais presentes |
| `README.md` completo | Aprovado com ressalva | Comandos estao documentados, mas treino falha ate corrigir dados/schema |
| `pyproject.toml` | Aprovado | Python 3.14.4 e scripts configurados |
| PyTorch MLP | Aprovado | Codigo presente e artefato local responde na API |
| Scikit-Learn baselines | Parcial | Codigo presente, mas comando falha ao carregar CSV real |
| MLflow | Parcial | Codigo presente, mas treino atual falha antes do registro com CSV real |
| FastAPI | Aprovado | `/predict` retornou 200 com artefatos atuais |
| Seeds fixados | Aprovado | `RANDOM_SEED = 42` |
| Validacao cruzada estratificada | Aprovado | Implementada em baseline e MLP |
| Model Card | Aprovado | Documento completo presente |
| Testes automatizados | Aprovado com ressalva | 9 testes passam, mas falta teste end-to-end de dados |
| Logging estruturado | Aprovado | `structlog` no core e sem `print()` em `src/churn` |
| Ruff sem erros | Aprovado | `ruff check .` passou |
| Ruff format | Reprovado | Dois arquivos de notebook precisam formatacao ou exclusao da regra |

## Recomendacoes Prioritarias

1. Corrigir `load_churn_csv` para limpar `TotalCharges` antes da validacao Pandera.
2. Adicionar `customerID` em `generate_sample_dataset`.
3. Criar teste end-to-end que rode `load_churn_csv` sobre CSV sintetico salvo em disco.
4. Criar teste com `TotalCharges` vazio para reproduzir o caso real do Kaggle.
5. Ajustar formatacao dos notebooks ou configurar exclusao consciente para notebooks.
6. Reexecutar:

```bash
uv run --no-editable --python 3.14.4 ruff check .
uv run --no-editable --python 3.14.4 ruff format --check .
uv run --no-editable --python 3.14.4 pytest
uv run --no-editable --python 3.14.4 churn-train-baseline --data data/raw/churn.csv
uv run --no-editable --python 3.14.4 churn-train-mlp --data data/raw/churn.csv --epochs 40
uv run --no-editable --python 3.14.4 churn-compare-models --data data/raw/churn.csv --epochs 40
```

## Status Final

Status atual: **nao aprovado para entrega final sem ajustes**.

Motivo: a documentacao e a API estao boas, mas o caminho principal de treino com o dataset real
falha na validacao de dados. Corrigindo os dois pontos de dados/schema e a formatacao indicada, o
projeto deve ficar muito proximo de compliance total.

