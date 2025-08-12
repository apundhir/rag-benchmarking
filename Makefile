help:
	@echo "Targets: env install run docker-up ingest-sample eval-golden test"

env:
	conda create -y -n rag_agentic python=3.11
	conda run -n rag_agentic python -m pip install -U pip
	conda run -n rag_agentic python -m pip install -e .

install:
	python -m pip install -U pip
	python -m pip install -e .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 5000

docker-up:
	HOST_PORT=5001 docker compose up -d

ingest-sample:
	python -m app.retrieval.ingest_cli data/sample/guide.md

eval-golden:
	set -a && source .env && set +a && \
	python scripts/evaluate.py data/golden/qa.jsonl --metrics faithfulness answer_relevancy --out reports/ragas_report.json

test:
	pytest -q
