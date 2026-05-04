# Arquitetura de Deploy e Monitoramento

## Visao Geral

A solucao foi desenhada para servir predicoes de churn em tempo real por meio de uma API REST com
FastAPI. O fluxo cobre treinamento local, rastreamento de experimentos, persistencia de artefatos e
inferencia online.

## Componentes

- `data/raw/churn.csv`: entrada com dados historicos de clientes.
- `src/churn/features.py`: pipeline de pre-processamento com Scikit-Learn.
- `src/churn/baseline.py`: baseline de regressao logistica.
- `src/churn/train.py`: treinamento da MLP PyTorch com validacao cruzada estratificada.
- `mlruns/` e `mlflow.db`: rastreamento local de experimentos.
- `models/`: artefatos treinados.
- `src/churn/api.py`: API FastAPI para inferencia.

## Deploy Sugerido

1. Treinar o modelo e gerar artefatos.
2. Empacotar a API em imagem Docker.
3. Publicar a imagem em um registry.
4. Servir em Cloud Run, AWS Fargate, ECS ou Kubernetes.
5. Configurar health check em `/health`.

## Monitoramento Recomendado

- Latencia da API.
- Taxa de erro HTTP.
- Distribuicao das probabilidades de churn.
- Drift das features de entrada.
- Queda de performance por janela temporal.
- Metricas por segmento de cliente.

## Riscos Operacionais

- Artefatos ausentes impedem inferencia.
- Dados de producao fora do schema esperado causam erro de validacao.
- Mudancas comerciais podem causar drift e reduzir performance.
