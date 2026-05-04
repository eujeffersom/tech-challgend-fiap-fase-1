FROM python:3.14-slim

# Metadados da imagem.
LABEL maintainer="Grupo 30"
LABEL version="0.1.0"
LABEL description="API de predicao de Churn de Telecom"

# Variaveis de ambiente para logs previsiveis e sem arquivos .pyc.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1

# Diretorio padrao da aplicacao.
WORKDIR /app

# Dependencias de sistema: curl para healthcheck e build-essential para libs Python nativas.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala o uv para respeitar pyproject.toml e uv.lock como fonte das dependencias.
RUN pip install --no-cache-dir uv

# Copia metadados primeiro para aproveitar cache de camadas das dependencias.
COPY pyproject.toml uv.lock README.md ./

# Instala apenas dependencias de producao antes de copiar o codigo.
RUN uv sync --frozen --no-dev --no-editable --no-install-project

# Copia o codigo fonte e os artefatos locais necessarios para a API.
COPY src/ ./src/
COPY models/ ./models/

# Instala o pacote do projeto apos o codigo fonte estar disponivel.
RUN uv sync --frozen --no-dev --no-editable

# Cria usuario nao-root e ajusta permissoes para reduzir risco no runtime.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Porta usada pela API FastAPI.
EXPOSE 8000

# Healthcheck usado por Docker, ECS, Fargate, Cloud Run ou orquestradores similares.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Sobe a API de inferencia em modo producao.
CMD ["uv", "run", "--no-sync", "uvicorn", "--app-dir", "src", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
