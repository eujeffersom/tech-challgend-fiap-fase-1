# ML Canvas

## Problema

Uma operadora de telecomunicacoes esta perdendo clientes e precisa identificar usuarios com alto
risco de churn para priorizar acoes de retencao.

## Objetivo de Negocio

Reduzir cancelamentos ao antecipar clientes em risco e apoiar campanhas de retencao mais precisas.

## Objetivo de Machine Learning

Classificar clientes entre risco baixo e alto de churn, retornando tambem a probabilidade estimada
de cancelamento.

## Usuarios

- Time de marketing.
- Time de relacionamento com cliente.
- Time de dados.
- Gestores de operacoes.

## Dados de Entrada

- Perfil do cliente.
- Tempo de contrato.
- Servicos contratados.
- Tipo de contrato.
- Metodo de pagamento.
- Cobrancas mensais e totais.

## Saida do Modelo

- `churn_probability`: probabilidade de churn.
- `churn_prediction`: classificacao binaria com threshold padrao de `0.5`.

## Metricas

- ROC-AUC.
- Recall.
- Precision.
- F1.
- Accuracy.

## Decisoes de Engenharia

- PyTorch para MLP.
- Scikit-Learn para pre-processamento e baseline.
- MLflow para rastreamento.
- FastAPI para servico de inferencia.
- Pytest e Ruff para qualidade.

## Riscos e Mitigacoes

- Drift de dados: monitorar distribuicoes e retreinar periodicamente.
- Vieses historicos: avaliar metricas por segmento.
- Threshold inadequado: calibrar conforme custo de falso positivo e falso negativo.
