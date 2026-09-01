# Convenience targets. On Windows, run the underlying commands directly or use `make` from Git Bash.
.PHONY: install dev ingest serve ask eval test lint fmt docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Build the vector index from data/corpus
ingest:
	gaprag ingest

serve:
	uvicorn gaprag.api:app --reload --port 8000

ask:
	gaprag ask "What is RAG and when should I use it?"

eval:
	python evals/run_eval.py

test:
	pytest

lint:
	ruff check src tests evals

fmt:
	ruff check --fix src tests evals

docker:
	docker build -t observable-rag .
