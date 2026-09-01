# Lean image: runs fully offline (lexical embeddings + mock LLM) so `docker run` just works.
# For real answers, use docker-compose (adds Ollama) or set provider env vars + keys.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir -e .

ENV GAPRAG_EMBEDDING_PROVIDER=hash \
    GAPRAG_LLM_PROVIDER=mock \
    GAPRAG_JSON_LOGS=1

EXPOSE 8000

# Build the index at startup, then serve.
CMD ["sh", "-c", "gaprag ingest && uvicorn gaprag.api:app --host 0.0.0.0 --port 8000"]
