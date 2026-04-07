FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./

RUN uv sync --no-dev

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
