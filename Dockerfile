# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Estagio 1: monta o .venv com as dependencias travadas no uv.lock
# ---------------------------------------------------------------------------
# Mesma versao do .python-version. Os dois estagios usam a mesma imagem para o
# Python do .venv existir no mesmo caminho na imagem final.
FROM python:3.14.4-slim AS build

# uv com versao fixa, direto da imagem oficial (antes era "pip install uv" sem versao).
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

# Compila os .pyc no build (a API sobe mais rapido) e usa o Python da imagem,
# sem baixar outro.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Primeiro so as dependencias: se mudar apenas o codigo, o Docker reaproveita essa camada.
# --no-default-groups deixa de fora treino (torch, mlflow...) e dev. A API nao usa nada disso.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable --no-install-project

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

# ---------------------------------------------------------------------------
# Estagio 2: imagem final, so com Python, o .venv e o modelo
# ---------------------------------------------------------------------------
FROM python:3.14.4-slim AS runtime

LABEL org.opencontainers.image.title="telco-churn-api" \
      org.opencontainers.image.description="API de predicao de churn de telecom" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.source="https://github.com/eujeffersom/tech-challgend-fiap-fase-1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    CHURN_PROJECT_ROOT=/app

# Usuario sem privilegio e sem shell de login. Os arquivos continuam do root,
# entao a API consegue ler o codigo e o modelo, mas nao alterar.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# O pacote ja esta instalado dentro do .venv, nao precisa copiar o src/.
COPY --from=build /app/.venv /app/.venv
COPY models/champion.joblib /app/models/champion.joblib

USER app

EXPOSE 8000

# Healthcheck com o proprio Python, assim nao precisa instalar curl na imagem.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
