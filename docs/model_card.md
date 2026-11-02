# Model Card - Previsao de Churn Telecom

## 1. Resumo

| Item | Valor |
| --- | --- |
| Tarefa | Classificacao binaria: o cliente vai cancelar (1) ou nao (0) |
| Modelo | Regressao Logistica (`class_weight="balanced"`) dentro de um `Pipeline` do Scikit-Learn |
| Pre-processamento | Mediana + `StandardScaler` nas numericas, moda + one-hot nas categoricas |
| Threshold | 0.58, escolhido na validacao cruzada pelo melhor F1 |
| Artefato | `models/champion.joblib` (pipeline + threshold, ~8 KB) |
| Servido por | `POST /predict` da API FastAPI |
| Versao | 0.2.0 |
| Dados de treino | Telco Customer Churn (Kaggle), 7.043 clientes, 26,5% de churn |

## 2. Uso pretendido

Priorizar clientes para acoes de retencao: quem tem mais risco recebe contato, oferta ou
atendimento primeiro. O score e um insumo para a decisao de um time (marketing, relacionamento,
atendimento), nao a decisao em si.

**Usos recomendados**

- Ordenar a base por risco para montar campanhas de retencao.
- Consultar o risco de um cliente no momento do atendimento, via API.
- Simular o impacto de thresholds diferentes no custo da campanha.

**Usos nao recomendados**

- Negar atendimento, desconto ou beneficio com base no score.
- Aplicar em outra operadora, outro pais ou outro periodo sem revalidar com dados de la.
- Interpretar os coeficientes como causa do churn (ver secao 6).

## 3. Dados

- Fonte: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- 7.043 clientes, 19 variaveis de entrada (perfil, servicos contratados, contrato, cobranca).
- `customerID` e removido antes do treino.
- 11 clientes com `TotalCharges` em branco, todos com `tenure = 0` (ainda sem a primeira
  cobranca). Sao imputados com a mediana.
- O CSV bruto e validado com Pandera antes de qualquer etapa.

Detalhes da analise exploratoria em `notebooks/01_eda_e_baselines.ipynb`.

## 4. Como o modelo foi escolhido

1. Split 80/20 estratificado. Os 20% de teste ficam guardados ate o fim.
2. Cinco candidatos (Regressao Logistica, Random Forest, Gradient Boosting, `MLPClassifier` e
   uma MLP em PyTorch) passam pela mesma validacao cruzada estratificada de 5 folds, dentro dos
   80% de treino.
3. Em cada fold o threshold com melhor F1 e escolhido no fold de validacao. O threshold final e
   a media dos folds.
4. Campeao: maior F1 medio na CV entre os modelos Scikit-Learn.
5. O campeao e retreinado com todo o treino e medido uma unica vez no teste.

| Modelo | ROC-AUC (CV) | PR-AUC (CV) | F1 (CV) | Recall (CV) | F1 (teste) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Regressao Logistica** | 0.8459 ± 0.0124 | 0.6600 ± 0.0195 | 0.6406 ± 0.0216 | 0.7411 ± 0.0521 | 0.6186 |
| Random Forest | 0.8415 ± 0.0081 | 0.6464 ± 0.0223 | 0.6375 ± 0.0115 | 0.7472 ± 0.0631 | 0.6274 |
| Gradient Boosting | 0.8478 ± 0.0122 | 0.6671 ± 0.0231 | 0.6374 ± 0.0166 | 0.7385 ± 0.0262 | 0.6257 |
| MLP PyTorch | 0.8458 ± 0.0126 | 0.6577 ± 0.0169 | 0.6366 ± 0.0241 | 0.7739 ± 0.0321 | 0.6222 |
| MLPClassifier | 0.8410 ± 0.0138 | 0.6549 ± 0.0213 | 0.6331 ± 0.0235 | 0.7144 ± 0.0694 | 0.6195 |

As diferencas entre os modelos sao menores que o desvio da propria validacao cruzada, entao na
pratica eles empatam. Com empate tecnico, a Regressao Logistica leva vantagem por ser mais
simples, mais rapida, ter um artefato de 8 KB (o Random Forest tem 21 MB) e permitir explicar
o resultado pelos coeficientes.

## 5. Performance no teste

1.409 clientes (20% da base), nunca vistos no treino nem na escolha do threshold.

| Metrica | Valor |
| --- | ---: |
| ROC-AUC | 0.8413 |
| PR-AUC | 0.6326 |
| Recall | 0.7219 |
| Precision | 0.5411 |
| F1 | 0.6186 |
| Accuracy | 0.7637 |

Na pratica: de cada 100 clientes que iam cancelar, o modelo encontra 72. De cada 100 clientes
que ele aponta como risco, 54 de fato cancelariam. Ele aciona 35% da base.

**Impacto financeiro estimado no teste**, com as premissas do `compare_models.py` (valor do
cliente de 1.000, 30% de sucesso na retencao, custo de 50 por acao):

| | Clientes |
| --- | ---: |
| Verdadeiros positivos (churn encontrado) | 270 |
| Falsos positivos (acao desnecessaria) | 229 |
| Falsos negativos (churn perdido) | 104 |
| Verdadeiros negativos | 806 |

Valor liquido estimado: **56.050** (churn evitado menos custo das acoes). As premissas sao
ilustrativas e precisam ser trocadas pelos numeros reais da operadora.

## 6. O que o modelo aprendeu

O one-hot mantem todas as categorias, entao o que importa e comparar as categorias de uma mesma
variavel entre si.

| Aumenta o risco | Diminui o risco |
| --- | --- |
| Internet por fibra otica | Mais tempo de casa (`tenure`, o maior peso) |
| Contrato mensal | Contrato de dois anos |
| Pagamento por boleto eletronico | Internet DSL ou sem internet |
| Nao ter `OnlineSecurity` nem `TechSupport` | Ter `OnlineSecurity` e `TechSupport` |
| Streaming de TV e filmes | Pagamento automatico (cartao ou debito) |

Bate com a EDA: contrato mensal, primeiro ano de casa, fibra e boleto eletronico concentram o
churn. Genero tem peso praticamente nulo.

Cuidado ao ler os coeficientes um por um. `TotalCharges` e quase exatamente
`tenure x MonthlyCharges` (correlacao 0.9996), e o preco mensal se confunde com o tipo de
internet e os servicos contratados. Por isso `MonthlyCharges` aparece com sinal negativo, mesmo
quem paga mais cancelando mais: o efeito do preco ja foi absorvido pela fibra e pelos servicos.
O modelo mostra associacao, nao causa.

## 7. Vieses e comportamento por segmento

Medido no conjunto de teste, com o threshold de producao (0.58).

| Segmento | Clientes | Churn real | Recall | Precision | Acionados |
| --- | ---: | ---: | ---: | ---: | ---: |
| Masculino | 722 | 25.1% | 0.72 | 0.52 | 35% |
| Feminino | 687 | 28.1% | 0.72 | 0.57 | 36% |
| Nao idoso | 1.187 | 23.3% | 0.68 | 0.52 | 30% |
| Idoso | 222 | 44.1% | 0.85 | 0.59 | 63% |
| Contrato mensal | 773 | 42.6% | 0.81 | 0.55 | 63% |
| Contrato anual | 300 | 12.0% | 0.11 | 0.36 | 4% |
| Contrato de dois anos | 336 | 2.7% | 0.00 | - | 0% |
| Ate 12 meses de casa | 449 | 47.9% | 0.80 | 0.62 | 62% |
| Mais de 48 meses de casa | 450 | 7.8% | 0.23 | 0.31 | 6% |

**Genero**: o modelo se comporta igual para homens e mulheres (mesmo recall e mesma taxa de
acionamento). A EDA ja mostrava que genero nao muda a taxa de churn.

**Idade**: idosos sao acionados duas vezes mais. Isso acompanha a taxa real de churn deles (44%
contra 23%), entao nao e um erro do modelo, mas tem consequencia pratica: se a acao de retencao
for algo indesejado (ligacoes insistentes, por exemplo), esse grupo recebe o dobro. Vale
acompanhar a experiencia desse publico nas campanhas.

**Ponto cego em clientes antigos e de contrato longo**: com o threshold atual, o modelo quase
nunca aponta clientes de contrato anual ou de dois anos. Entre clientes com mais de 4 anos de
casa, encontra so 23% dos que cancelam. Sao poucos cancelamentos, mas costumam ser os clientes
de maior valor. Se isso importar para o negocio, da para usar um threshold menor nesses
segmentos ou uma acao especifica para eles.

**Proxies**: forma de pagamento e tipo de contrato podem refletir renda ou acesso a servico
bancario. Antes de usar em producao, vale revisar se as acoes geradas pelo modelo tratam esses
grupos de forma justa.

## 8. Limitacoes

- Dataset publico e historico, de uma unica operadora. Precos, produtos e concorrencia mudam.
- So enxerga o que esta no cadastro: nao tem historico de atendimento, reclamacoes, uso da rede
  nem ofertas da concorrencia, que costumam pesar muito no churn.
- O threshold foi escolhido para equilibrar F1. Outro objetivo de negocio (ex.: maximizar valor
  liquido) pode pedir outro corte.
- As probabilidades nao foram calibradas. Servem bem para ordenar clientes, mas "0.80" nao quer
  dizer exatamente 80% de chance.
- O `.joblib` depende da versao do Scikit-Learn usada no treino. Treino e API usam o mesmo
  `uv.lock` para isso nao quebrar.

## 9. Cenarios de falha

| Cenario | O que acontece | Protecao atual |
| --- | --- | --- |
| Campo faltando ou com tipo errado no payload | Nao ha predicao | API responde 422 explicando o campo |
| Categoria nova (ex.: novo tipo de contrato) | Nao ha predicao | API responde 422; o modelo precisa ser retreinado |
| Modelo ausente | Nao ha predicao | API responde 503 e o `/health` continua ok |
| Mudanca no perfil dos clientes (drift) | Predicoes piores sem erro visivel | Monitoramento (`docs/monitoring_plan.md`) |
| Mudanca de custo da campanha | Threshold deixa de ser o melhor | Revisar threshold com o time de negocio |

## 10. Manutencao

- Retreinar com `make compare-models` quando houver dados novos rotulados, drift relevante ou
  mudanca de produto/preco. O comando reavalia todos os candidatos e pode trocar o campeao.
- A API recarrega o `champion.joblib` sozinha quando o arquivo muda.
- Plano de monitoramento e alertas em `docs/monitoring_plan.md`.
