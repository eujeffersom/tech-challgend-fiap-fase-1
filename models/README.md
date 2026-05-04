# Modelos

Artefatos treinados sao salvos localmente nesta pasta apos executar os comandos de treino.

## Artefatos principais

- `churn_mlp.pt`: checkpoint PyTorch da MLP principal, incluindo pesos e threshold.
- `preprocessor.joblib`: pre-processador Scikit-Learn usado pela MLP e pela API.
- `baseline_logreg.joblib`: baseline de Regressao Logistica gerada por `churn-train-baseline`.

## Artefatos da comparacao de modelos

Gerados por `churn-compare-models`:

- `logistic_regression.joblib`: Regressao Logistica usada na tabela comparativa.
- `random_forest.joblib`: Random Forest usada na tabela comparativa.
- `gradient_boosting.joblib`: Gradient Boosting usado na tabela comparativa.

## Observacoes

- Os arquivos `.pt`, `.joblib` e `.pkl` sao ignorados pelo Git por padrao.
- Para recriar os artefatos, execute os comandos de treino no `README.md` da raiz.
- A API espera encontrar `churn_mlp.pt` e `preprocessor.joblib` neste diretorio.
