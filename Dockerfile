FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    SERVE_WEB=1 \
    WEB_DIST=/app/web/dist

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i "$PIP_INDEX_URL" .

COPY web/dist ./web/dist
COPY knowledge ./knowledge
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "research_team.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
