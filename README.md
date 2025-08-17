# Receiper

Recipe Q&A and cooking assistance with Retrieval-Augmented Generation (RAG). Answers come with citations from a curated recipe corpus.

## Quickstart

1. Copy `.env.example` to `.env` and fill in keys if using OpenAI.
2. Create a venv and install deps:
   ```bash
   make setup
   ```
3. Run services via Docker:
   ```bash
   make up
   ```
4. Start API and UI locally (alternative):
   ```bash
   make run-api
   make run-ui
   ```

## Docs
- See `docs/PROJECT_PLAN.md` for the full plan.
- Setup, usage, evaluation, and monitoring docs will be added under `docs/`.
