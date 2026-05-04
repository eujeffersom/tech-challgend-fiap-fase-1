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
preprocessor e o mesmo modelo versionado.

## 3. Componentes

- `data/raw/churn.csv`: base historica de treino.
- `src/churn/schema.py`: validacao de schema com Pandera.
- `src/churn/features.py`: pipeline de pre-processamento com Scikit-Learn.
- `src/churn/baseline.py`: baseline de regressao logistica.
- `src/churn/compare_models.py`: comparacao entre modelo linear, arvores, ensemble e MLP.
- `src/churn/train.py`: treinamento da MLP PyTorch com batching, early stopping e validacao cruzada.
- `mlruns/`: tracking local de experimentos no MLflow.
- `models/`: artefatos locais do modelo.
- `src/churn/api.py`: API FastAPI com `/health`, `/predict`, Pydantic, logging estruturado e middleware de latencia.
- `Dockerfile`: empacotamento da API para deploy.

## 4. Fluxo de Treinamento

1. Baixar o dataset Telco Customer Churn e salvar em `data/raw/churn.csv`.
2. Validar schema e regras basicas com Pandera.
3. Separar treino e teste com estratificacao.
4. Ajustar preprocessor no treino.
5. Treinar baselines e MLP.
6. Registrar parametros, metricas e artefatos no MLflow.
7. Persistir artefatos em `models/`.
8. Gerar tabela comparativa em `docs/reports/model_comparison.md`.

## 5. Fluxo de Inferencia

1. Cliente interno chama `POST /predict`.
2. Pydantic valida o payload da requisicao.
3. API aplica o mesmo preprocessor salvo no treino.
4. MLP calcula o logit e a probabilidade de churn.
5. API aplica o threshold salvo/configurado.
6. Resposta retorna classe prevista, probabilidade e threshold.
7. Middleware registra latencia e logs estruturados.

## 6. Deploy Sugerido

1. Treinar o modelo e gerar artefatos.
2. Executar testes e lint.
3. Buildar a imagem Docker.
4. Publicar a imagem em um registry.
5. Subir a API em Cloud Run, AWS Fargate, ECS ou Kubernetes.
6. Configurar health check em `/health`.
7. Configurar logs, metricas e alertas.
8. Versionar modelo, imagem e parametros de threshold.

Comandos locais:

```bash
uv run --no-editable --python 3.14.4 churn-train-mlp --data data/raw/churn.csv --epochs 40
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
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

