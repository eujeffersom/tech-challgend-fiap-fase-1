# Arquitetura de Deploy e Monitoramento

## 1. Decisao Arquitetural

A arquitetura escolhida para a entrega principal e **inferencia em tempo real via REST API**. O
modelo e servido por FastAPI, empacotado em Docker e pronto para rodar em servicos como Cloud Run,
AWS Fargate, ECS ou Kubernetes.

Essa escolha foi feita porque churn exige integracao com sistemas de relacionamento e atendimento:
um CRM, call center ou painel comercial pode consultar o score no momento em que o cliente entra em
uma jornada de retencao.

## 2. Real-Time vs. Batch

| Opcao | Uso | Vantagem | Limitacao |
| --- | --- | --- | --- |
| Real-time API | Score sob demanda por cliente | Integra facil com CRM, app interno e jornadas online | Exige disponibilidade e monitoramento da API |
| Batch scoring | Score periodico de toda a base | Bom para campanhas massivas e listas diarias | Menos adequado para decisao no momento do atendimento |

Escolha da entrega: **real-time como caminho principal**.

Batch pode ser adicionado como complemento futuro para campanhas diarias ou semanais, usando o mesmo
`champion.joblib` que a API carrega.

## 3. Componentes

- `data/raw/churn.csv`: base historica de treino.
- `src/churn/schema.py`: validacao de schema com Pandera.
- `src/churn/features.py`: pipeline de pre-processamento com Scikit-Learn.
- `src/churn/baseline.py`: baseline de regressao logistica.
- `src/churn/compare_models.py`: compara os candidatos com a mesma validacao cruzada e salva o
  campeao em `models/champion.joblib`.
- `src/churn/train.py`: MLP PyTorch, mantida como experimento extra.
- `mlflow.db`: tracking local dos experimentos no MLflow.
- `models/champion.joblib`: pipeline campeao + threshold. E o unico artefato que vai para o Git
  e para a imagem Docker.
- `src/churn/api.py` e `src/churn/api_models.py`: API FastAPI com `/health`, `/predict`,
  validacao do payload, logging estruturado e middleware de latencia.
- `Dockerfile`: imagem em dois estagios, so com as dependencias da API e o modelo campeao.

## 4. Fluxo de Treinamento

1. Baixar o dataset Telco Customer Churn e salvar em `data/raw/churn.csv`.
2. Validar schema e regras basicas com Pandera.
3. Separar treino e teste com estratificacao.
4. Rodar a mesma validacao cruzada para todos os candidatos, com o pre-processamento ajustado
   dentro de cada fold.
5. Escolher o threshold e o campeao so com a validacao cruzada.
6. Medir o campeao uma vez no teste.
7. Registrar parametros, metricas e artefatos no MLflow.
8. Salvar `models/champion.joblib` e gerar `docs/reports/model_comparison.md`.

## 5. Fluxo de Inferencia

1. Cliente interno chama `POST /predict`.
2. Pydantic valida o payload da requisicao.
3. API usa o pipeline campeao, que ja traz o pre-processamento ajustado no treino.
4. O modelo calcula a probabilidade de churn.
5. API aplica o threshold salvo junto com o modelo.
6. Resposta retorna classe prevista, probabilidade e threshold.
7. Middleware registra latencia e logs estruturados.

## 6. Deploy Sugerido

1. Rodar `make compare-models` e versionar o `champion.joblib` gerado.
2. Executar `make check`.
3. Buildar a imagem Docker.
4. Publicar a imagem em um registry.
5. Subir a API em Cloud Run, AWS Fargate, ECS ou Kubernetes.
6. Configurar health check em `/health`.
7. Configurar logs, metricas e alertas.
8. Versionar modelo, imagem e parametros de threshold.

Comandos locais:

```bash
make compare-models
make docker-build
make docker-run
```

URL local da API:

```text
http://127.0.0.1:8000/docs
```

## 7. Requisitos Operacionais

- Disponibilidade alvo: 99% para ambiente de demonstracao ou homologacao.
- Latencia alvo: p95 abaixo de 300 ms em requisicoes individuais.
- Health check: `/health`.
- Logs: JSON estruturado com evento, nivel e timestamp.
- Versionamento: registrar versao do codigo, dataset, parametros e artefatos no MLflow.
- Segurança: restringir acesso da API em producao por rede privada, API gateway ou autenticacao.

## 8. Riscos Operacionais

- Artefatos ausentes impedem a API de inferir.
- Payload fora do schema causa erro de validacao.
- Categorias novas podem gerar comportamento inesperado se nao forem monitoradas.
- Drift de dados ou conceito pode reduzir performance.
- Threshold antigo pode gerar custo excessivo quando a campanha muda.

O plano detalhado de monitoramento e resposta esta em `docs/monitoring_plan.md`.

