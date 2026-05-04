# Model Card - Previsao de Churn Telecom

## 1. Resumo do Modelo

Este projeto entrega um classificador de churn para clientes de telecomunicacoes. O modelo central
da entrega e uma rede neural MLP treinada com PyTorch, comparada com baselines e ensembles em
Scikit-Learn, rastreada no MLflow e servida por uma API FastAPI.

- Caso de uso: priorizar clientes com maior risco de cancelamento.
- Tipo de problema: classificacao binaria.
- Classe positiva: cliente com churn.
- Modelo principal: `pytorch_mlp`.
- Artefatos principais: `models/churn_mlp.pt` e `models/preprocessor.joblib`.
- Pipeline de features: `ColumnTransformer` do Scikit-Learn com tratamento de variaveis numericas e categoricas.

## 2. Uso Pretendido

O modelo deve apoiar times de retencao, relacionamento, marketing e operacoes comerciais na
priorizacao de campanhas. A predicao deve ser usada como insumo para acao humana ou regra de
negocio, nao como decisao automatica unica.

Usos recomendados:

- Ordenar clientes por risco de churn para campanhas de retencao.
- Simular impacto de thresholds diferentes no custo de campanha.
- Apoiar dashboards de risco por carteira, contrato ou canal.
- Integrar score de churn em uma API transacional para sistemas internos.

Usos nao recomendados:

- Negar atendimento, suporte ou beneficio essencial automaticamente.
- Tomar decisoes sensiveis usando apenas o score do modelo.
- Aplicar o modelo em bases com schema, pais, produto ou periodo muito diferentes do treino sem revalidacao.

## 3. Dados

O dataset usado e o Telco Customer Churn.

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Arquivo original: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Caminho esperado no projeto: `data/raw/churn.csv`
- Alvo esperado: `Churn`

O dataset contem atributos de contrato, servicos, cobranca e perfil do cliente. Antes do treino, a
base passa por validacao de schema com Pandera para checar colunas esperadas, tipos basicos e regras
de dominio, por exemplo `tenure >= 0`, `MonthlyCharges >= 0` e alvo em `Yes` ou `No`.

## 4. Preparacao e Treinamento

O treinamento usa uma separacao estratificada entre treino e teste, preservando a proporcao da
classe churn. A MLP tambem usa validacao cruzada estratificada para selecionar threshold e acompanhar
estabilidade.

Boas praticas aplicadas:

- Seed global fixa em `42`.
- Split treino/teste estratificado.
- Validacao cruzada estratificada.
- Pre-processamento ajustado apenas no treino.
- Batching com `DataLoader`.
- Early stopping por perda de validacao.
- `BCEWithLogitsLoss` com `pos_weight` para reduzir impacto do desbalanceamento.
- Threshold de classificacao selecionado por validacao, em vez de assumir `0.5`.
- Registro de parametros, metricas e artefatos no MLflow.

## 5. Performance

Resultado comparativo registrado em `docs/reports/model_comparison.md`.

| Modelo | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Valor liquido estimado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.45 | 0.7665 | 0.5442 | 0.7406 | 0.6274 | 0.8378 | 0.6345 | 57650 |
| Gradient Boosting | 0.35 | 0.7764 | 0.5626 | 0.7086 | 0.6272 | 0.8434 | 0.6641 | 55950 |
| PyTorch MLP | 0.53 | 0.7580 | 0.5313 | 0.7487 | 0.6215 | 0.8427 | 0.6335 | 57650 |
| Logistic Regression | 0.55 | 0.7530 | 0.5243 | 0.7513 | 0.6176 | 0.8413 | 0.6326 | 57500 |

Leitura dos resultados:

- Acuracia isolada nao e suficiente, pois a base tem desbalanceamento de churn.
- Recall, F1, ROC-AUC e PR-AUC sao mais relevantes para avaliar captura da classe de cancelamento.
- A MLP ficou competitiva com baselines e ensembles, com recall alto e ROC-AUC proximo do melhor modelo.
- Random Forest apresentou melhor F1 na rodada registrada, mas a MLP permanece como modelo central da entrega por requisito do projeto.

## 6. Metricas de Negocio

O projeto mede um trade-off simples de custo:

- Falso positivo: cliente sem churn recebe acao de retencao desnecessaria.
- Falso negativo: cliente com churn nao recebe acao e pode cancelar.
- Valor liquido estimado: valor de churn evitado menos custo de acionamento.

A formula e parametrizavel no codigo para adaptar valores de campanha e perda esperada. Em producao,
o threshold deve ser escolhido com o time de negocio, porque o melhor threshold tecnico pode nao ser
o melhor threshold financeiro.

## 7. Limitacoes

- O dataset e historico e pode nao refletir campanhas, precos ou comportamento atual.
- O modelo depende da consistencia das colunas de entrada e dos valores categoricos.
- Clientes novos, com pouco tempo de contrato, podem ter historico insuficiente.
- O modelo nao explica causalidade; ele estima risco com base em associacoes historicas.
- O threshold atual foi escolhido com validacao offline e deve ser recalibrado apos observar dados reais.
- A performance registrada vem de um dataset publico; antes de producao real, e preciso validar com dados da operadora.

## 8. Vieses e Riscos

Possiveis fontes de vies:

- Variaveis demograficas podem refletir historico desigual de ofertas, suporte ou precificacao.
- Segmentos com poucos exemplos podem ter estimativas menos confiaveis.
- Categorias de contrato ou pagamento podem ser proxies para renda, canal ou perfil regional.
- Mudancas comerciais podem favorecer ou prejudicar grupos especificos sem que o modelo perceba imediatamente.

Mitigacoes recomendadas:

- Monitorar precision, recall e taxa de acionamento por segmento.
- Evitar uso automatizado para negar beneficios ou atendimento.
- Revisar features sensiveis ou proxies antes de producao.
- Validar campanhas de retencao com amostras de controle.
- Documentar qualquer alteracao de dados, features, threshold ou politica comercial.

## 9. Cenarios de Falha

- Schema inesperado: coluna ausente ou tipo errado causa erro de validacao.
- Drift de dados: distribuicao de cobrancas, contratos ou servicos muda apos reajustes comerciais.
- Drift de conceito: o comportamento de churn muda por concorrencia, atendimento ou campanha.
- Artefatos ausentes: API nao consegue carregar `preprocessor` ou pesos da MLP.
- Threshold inadequado: custo de campanha muda e o threshold antigo passa a gerar prejuizo.
- Baixa calibracao: probabilidade prevista pode nao representar probabilidade real em todos os segmentos.
- Latencia ou indisponibilidade: sistemas consumidores podem falhar ao chamar `/predict`.

## 10. Monitoramento e Atualizacao

O modelo deve ser monitorado em tres camadas:

- Operacional: latencia, taxa de erro, disponibilidade e throughput da API.
- Dados: schema, nulos, categorias novas, drift de features e distribuicao dos scores.
- Modelo/negocio: recall, precision, PR-AUC, churn evitado, custo de campanha e taxa de conversao.

O retreinamento deve ser acionado quando houver drift relevante, queda de performance, mudanca de
produto/preco ou nova base rotulada suficiente para revalidacao.

