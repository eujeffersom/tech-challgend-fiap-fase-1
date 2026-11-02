# ML Canvas

## Problema

Uma operadora de telecomunicacoes esta perdendo clientes e precisa identificar usuarios com alto
risco de churn para priorizar acoes de retencao.

## Stakeholders

- Diretoria: acompanha reducao de churn e impacto financeiro.
- Marketing: cria campanhas de retencao para clientes em risco.
- Atendimento e relacionamento: prioriza contatos preventivos.
- Time de dados: treina, monitora e melhora os modelos.
- Engenharia/Plataforma: disponibiliza API, observabilidade e deploy.

## Objetivo de Negocio

Reduzir cancelamentos ao antecipar clientes em risco e apoiar campanhas de retencao mais precisas.

## Metricas de Negocio

- Taxa de churn mensal.
- Receita preservada por campanha de retencao.
- Custo de churn evitado.
- Taxa de conversao das acoes de retencao.
- Custo por cliente abordado.

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
- `churn_prediction`: classificacao binaria conforme threshold selecionado no treino.

## Metricas Tecnicas

- **F1 (principal)**: e o criterio de escolha do modelo campeao e do threshold. Equilibra achar
  quem vai cancelar (recall) com nao gastar campanha em quem ia ficar (precision).
- **Recall** e **PR-AUC**: acompanham a classe de churn, que e a minoria (26,5%).
- **ROC-AUC**: mede a capacidade de ordenar clientes por risco, independente do threshold.
- **Accuracy**: so para referencia. Um modelo que diz "ninguem sai" ja tem 73,5%.

Todas sao medidas com validacao cruzada estratificada de 5 folds (media e desvio) e confirmadas
uma unica vez no conjunto de teste.

## Metrica de Negocio Principal

O projeto usa **custo de churn evitado** como metrica de negocio. A ideia e comparar o valor
financeiro preservado ao identificar clientes que realmente cancelariam contra o custo de abordar
clientes que nao cancelariam.

Formula simplificada:

```text
custo_churn_evitado = (TP * valor_cliente * taxa_sucesso_retencao) - ((TP + FP) * custo_acao)
```

Onde:

- `TP`: clientes churn corretamente priorizados.
- `FP`: clientes abordados que nao iriam cancelar.
- `valor_cliente`: valor medio estimado do cliente.
- `taxa_sucesso_retencao`: percentual de clientes retidos apos acao.
- `custo_acao`: custo medio de abordagem/beneficio.

## SLOs

- Latencia p95 da API de predicao abaixo de 300 ms em ambiente de producao.
- Disponibilidade mensal da API maior ou igual a 99%.
- Taxa de erro 5xx abaixo de 1%.
- Retreino ou revisao mensal, ou antes em caso de drift relevante.
- Monitoramento semanal de distribuicao das features e das probabilidades.

## Criterios de Sucesso

- Superar o baseline Dummy em ROC-AUC, recall e F1. Atingido: o Dummy tem F1 0 e ROC-AUC 0.50;
  o campeao tem F1 0.64 e ROC-AUC 0.85 na validacao cruzada.
- Encontrar a maior parte de quem vai cancelar. Atingido: recall de 0.72 no teste.
- Escolher o modelo com o mesmo processo para todos os candidatos e registrar tudo no MLflow.

## Decisoes de Engenharia

- Scikit-Learn para pre-processamento e para todos os modelos candidatos (Regressao Logistica,
  Random Forest, Gradient Boosting e `MLPClassifier`). O campeao atual e a Regressao Logistica.
- MLP em PyTorch mantida como experimento extra de comparacao.
- MLflow para rastreamento dos experimentos.
- FastAPI para o servico de inferencia, com o payload validado por Pydantic.
- Pytest e Ruff para qualidade, Docker para deploy.

## Riscos e Mitigacoes

- Drift de dados: monitorar distribuicoes e retreinar periodicamente.
- Vieses historicos: avaliar metricas por segmento.
- Threshold inadequado: calibrar conforme custo de falso positivo e falso negativo.
