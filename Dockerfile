FROM python:3.14.4-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY models ./models

RUN uv sync --frozen --no-dev --no-editable

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "--app-dir", "src", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
