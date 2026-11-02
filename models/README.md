# Modelos

## O que vai para o Git

- `champion.joblib`: modelo que a API serve. E um dicionario com o pipeline completo
  (pre-processamento + classificador), o threshold escolhido na validacao cruzada e o nome
  do modelo. Fica versionado para a API e o Docker funcionarem logo depois do clone.

Hoje o campeao e a Regressao Logistica. O criterio e o maior F1 medio na validacao cruzada
entre os modelos Scikit-Learn (ver `docs/reports/model_comparison.md`).

## O que fica so na maquina

Gerados por `churn-compare-models` e ignorados pelo Git:

- `logistic_regression.joblib`, `random_forest.joblib`, `gradient_boosting.joblib`,
  `mlp_classifier.joblib`: cada candidato da comparacao, ja treinado.
- `churn_mlp.pt` e `preprocessor.joblib`: MLP PyTorch, mantida como experimento extra.

Gerado por `churn-train-baseline`:

- `baseline_logreg.joblib`: baseline de Regressao Logistica.

## Como recriar

```bash
make compare-models
```

Isso retreina tudo e sobrescreve o `champion.joblib`. Se o ranking mudar, o campeao muda junto.

## Cuidado com versoes

O `.joblib` depende da versao do Scikit-Learn usada no treino. Use sempre o ambiente do
`uv.lock` (`make setup`) para treinar e para rodar a API.
