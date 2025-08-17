PY=python3
PIP=pip

.PHONY: setup run-api run-ui ingest up down eval fmt lint

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run app/ui/streamlit_app.py --server.port 8501

ingest:
	$(PY) ingestion/ingest_openrecipes.py download --out data/raw && \
	$(PY) ingestion/ingest_openrecipes.py process --inp data/raw --out data/processed && \
	$(PY) ingestion/ingest_openrecipes.py index --chunks data/processed/chunks.parquet

up:
	docker compose -f docker/docker-compose.yml up -d --build

down:
	docker compose -f docker/docker-compose.yml down -v

eval:
	$(PY) eval/retrieval_eval.py --grid default && $(PY) eval/llm_eval_ragas.py

fmt:
	ruff check --fix . || true && black . && isort . || true

lint:
	ruff check .
