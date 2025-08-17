#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python ingestion/ingest_openrecipes.py download --out data/raw
python ingestion/ingest_openrecipes.py process --inp data/raw --out data/processed --chunk-tokens 500 --overlap 100
python ingestion/ingest_openrecipes.py index --chunks data/processed/chunks.parquet

uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
