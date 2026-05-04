# Plano de Monitoramento

## 1. Objetivo

Monitorar a API, os dados, o modelo e o resultado de negocio para garantir que a solucao de churn
continue confiavel apos o deploy. O monitoramento deve detectar falhas operacionais, mudancas na
distribuicao dos dados e queda de performance do modelo.

## 2. Metricas Operacionais

| Metrica | Como medir | Alerta sugerido |
| --- | --- | --- |
| Disponibilidade | Health check em `/health` | API indisponivel por 5 minutos |
| Latencia p95 | Header `X-Process-Time-ms` e APM | p95 acima de 300 ms por 10 minutos |
| Taxa de erro 5xx | Logs/API gateway | acima de 1% por 10 minutos |
| Taxa de erro 4xx | Logs/API gateway | aumento brusco ou acima de 5% |
| Throughput | Requisicoes por minuto | queda inesperada em horario comercial |

## 3. Metricas de Dados

| Metrica | Como medir | Alerta sugerido |
| --- | --- | --- |
| Schema valido | Validacao Pandera/Pydantic | qualquer coluna obrigatoria ausente |
| Nulos por coluna | Perfilamento dos payloads | aumento acima de 5 p.p. |
| Categorias novas | Comparar com treino | categoria nova em feature critica |
| Drift numerico | PSI ou KS test | PSI acima de 0.20 |
| Drift categorico | PSI ou variacao de proporcao | PSI acima de 0.20 |
| Distribuicao dos scores | Histograma de probabilidade | deslocamento relevante por 7 dias |

## 4. Metricas de Modelo

As metricas de modelo dependem de rotulo real de churn, que normalmente chega com atraso. Quando os
rotulos estiverem disponiveis, acompanhar:

- ROC-AUC.
- PR-AUC.
- Precision.
- Recall.
- F1.
- Matriz de confusao.
- Calibration drift.
- Performance por segmento: contrato, internet, metodo de pagamento, senioridade e tenure.

Alertas sugeridos:

- Recall cair mais de 10% em relacao ao baseline validado.
- PR-AUC cair mais de 10% em janela mensal.
- Taxa de clientes classificados como alto risco dobrar sem explicacao operacional.
- Segmento especifico ter queda forte de precision ou recall.

## 5. Metricas de Negocio

- Churn evitado estimado.
- Custo de acionamento de retencao.
- Conversao da campanha de retencao.
- Receita preservada.
- Taxa de falso positivo operacional: clientes acionados sem churn.
- Taxa de falso negativo operacional: clientes com churn nao acionados.

O threshold deve ser revisado sempre que o custo da campanha, o valor do cliente ou a estrategia de
retencao mudar.

## 6. Playbook de Resposta

### Alerta de API indisponivel

1. Verificar `/health` e logs recentes.
2. Confirmar se a imagem Docker subiu com os artefatos em `models/`.
3. Validar variaveis de ambiente e permissao de leitura dos artefatos.
4. Fazer rollback para a ultima imagem saudavel se a falha for de release.
5. Registrar incidente e causa raiz.

### Alerta de schema ou erro 4xx

1. Coletar exemplos anonimizados de payloads com falha.
2. Comparar payload com `PredictionRequest` e schema de treino.
3. Confirmar se houve mudanca no sistema origem.
4. Corrigir contrato de dados ou adicionar transformacao controlada.
5. Criar teste automatizado para o caso novo.

### Alerta de drift

1. Identificar features com maior desvio.
2. Verificar eventos comerciais recentes: reajuste, campanha, novo produto ou canal.
3. Comparar performance por segmento quando houver rotulo.
4. Recalibrar threshold se a distribuicao de score mudou.
5. Retreinar modelo se houver dados rotulados suficientes.

### Alerta de queda de performance

1. Validar qualidade dos rotulos recentes.
2. Comparar matriz de confusao atual com a versao validada.
3. Avaliar se o custo de falso positivo/falso negativo mudou.
4. Rodar `churn-compare-models` com dados atualizados.
5. Promover nova versao apenas se superar criterios tecnicos e de negocio.

## 7. Cadencia de Revisao

- Diario: saude da API, erros, latencia e volume.
- Semanal: distribuicao dos scores, drift de features e taxa de acionamento.
- Mensal: performance com rotulos, metricas por segmento e custo de campanha.
- A cada mudanca comercial relevante: revisar threshold e necessidade de retreinamento.

