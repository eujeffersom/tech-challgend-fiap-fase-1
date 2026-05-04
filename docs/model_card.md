# Model Card - Previsao de Churn Telecom

## Visao Geral

Este modelo estima a probabilidade de churn de clientes de telecomunicacoes a partir de atributos
contratuais, uso de servicos e cobranca. O modelo principal e uma rede neural MLP treinada com
PyTorch e comparada com uma baseline de regressao logistica em Scikit-Learn.

## Uso Pretendido

- Priorizacao de acoes de retencao.
- Analise de risco por cliente.
- Apoio a times de marketing, relacionamento e sucesso do cliente.

O modelo nao deve ser usado como unica fonte para decisoes comerciais sensiveis.

## Dados

Entrada esperada:

- Atributos categoricos: genero, contrato, metodo de pagamento, servicos contratados.
- Atributos numericos: tempo de contrato, cobranca mensal, cobranca total e indicador senior.
- Alvo: `Churn`, com valores `Yes` ou `No`.

## Metricas

As metricas rastreadas no MLflow incluem:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Para churn, `recall` e `ROC-AUC` tendem a ser mais importantes que accuracy, pois a classe de
cancelamento pode ser minoritaria.

## Limitacoes

- O modelo depende da qualidade e atualidade dos dados historicos.
- Mudancas em precos, campanhas ou atendimento podem causar drift.
- Dados sinteticos servem apenas para validacao tecnica do pipeline, nao para avaliacao real.
- O threshold padrao `0.5` pode nao otimizar custo de retencao.

## Vieses e Riscos

- Variaveis demograficas podem refletir vieses historicos de atendimento ou oferta.
- Clientes com pouco historico podem ter previsoes menos confiaveis.
- Segmentos sub-representados no treino podem receber scores menos calibrados.

## Mitigacoes Recomendadas

- Monitorar metricas por segmento.
- Recalibrar threshold conforme custo de falso positivo e falso negativo.
- Retreinar periodicamente.
- Monitorar drift de features e queda de performance em producao.
