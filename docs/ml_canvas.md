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

- ROC-AUC.
- PR-AUC.
- Recall.
- Precision.
- F1.
- Accuracy.

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

- Modelo superar baseline Dummy em ROC-AUC, recall e F1.
- Modelo ser comparavel ou superior a regressao logistica em ROC-AUC.
- Recall adequado para priorizar clientes em risco sem depender apenas de accuracy.
- Experimentos registrados no MLflow com parametros, metricas e versao do dataset.

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
