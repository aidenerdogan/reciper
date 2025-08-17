# Receiper — End‑to‑End RAG for Recipe Q&A and Cooking Assistance

A simple, useful RAG app: ask cooking questions, get grounded answers with citations from a recipe corpus. Includes hybrid search, re-ranking, query rewriting, monitoring, and full documentation.

Last updated: 2025-08-17

---

## 1) Problem Description

Home cooks and food enthusiasts are overwhelmed by scattered, inconsistent cooking advice across blogs, videos, and social posts. They need trustworthy, context-aware guidance that respects constraints (ingredients on hand, dietary needs, time, and skill level) and provides verifiable sources. Traditional search often returns long articles without direct answers; generic LLM chat can hallucinate techniques or unsafe substitutions without citations.

Receiper solves this by turning a curated recipe corpus into a searchable knowledge base and layering a Retrieval-Augmented Generation (RAG) pipeline on top. Users ask natural-language questions like “30‑minute vegan dinner with chickpeas and spinach,” “How do I temper chocolate?” or “What can replace buttermilk in pancakes?” The system retrieves the most relevant recipe passages and technique notes, then composes a concise, grounded answer with clickable citations back to the sources.

Who it is for:
- Beginners needing step-by-step technique explanations with safety in mind.
- Busy home cooks filtering recipes by time, ingredients, and dietary constraints.

What problems it solves:
- Finds recipes that match multi-constraint queries (ingredients/diet/time).
- Explains cooking techniques and tricky steps with evidence from sources.
- Suggests safe substitutions and highlights trade-offs, with citations.

Why RAG (and not just an LLM):
- Reduces hallucinations by constraining answers to retrieved passages.
- Provides source transparency for users to validate claims.
- Hybrid search (lexical + vector) and re-ranking improve retrieval precision; query rewriting helps with vague or shorthand user questions.

Scope (MVP):
- Recipe discovery under constraints; technique Q&A; substitution guidance.
- Chat UI (Streamlit) and REST API (FastAPI) with citations in every answer.
- Logging and feedback collection to iteratively improve retrieval and prompts.

Non-goals (v1):
- Full meal planning/nutrition tracking, automated grocery lists, large-scale web crawling beyond the curated dataset.

Success criteria:
- High perceived relevance and faithfulness (user feedback ≥80% positive on evaluated sessions).
- Correct, clickable citations to source passages for every answer.
- Measurable retrieval quality (e.g., Recall@10 ≥ target on held-out queries) and acceptable latency for interactive use.

---

## 2) Dataset

- Primary source: Open Recipes (~170k recipes, JSON).
  - Repo: https://github.com/fictivekin/openrecipes (publicly available; see repository for licensing terms).
- Optional add-ons:
  - User-uploaded PDFs/URLs (personal recipe files) for private enrichment.
  - A small synthetic Q/A set for evaluation (generated programmatically).

What we ingest and keep (schema):
- id: stable document id (hash of source URL + title).
- title: recipe title.
- url: canonical source URL (for citations).
- ingredients: normalized list of ingredients (quantity, unit, item).
- instructions: cleaned text of steps (may be list -> joined string).
- tags: cuisine, diet, method when available.
- time: prep/cook/total minutes if present.
- yields: servings if present.

Cleaning and normalization:
- Strip HTML, normalize unicode/whitespace.
- Standardize units ("tsp"/"teaspoon", "g"/"grams") when unambiguous.
- Lowercase ingredients for matching; preserve original casing in text.
- Remove obviously corrupted or empty entries.
- Deduplicate near-identical recipes by URL/title shingle similarity.

Chunking strategy (for retrieval):
- instructions: semantic chunks of ~500 tokens with 100-token overlap.
- ingredients: a separate short chunk per recipe for constraint filtering.
- metadata-only chunk: title + tags to assist lexical match.
Each chunk carries metadata {doc_id, title, url, section, position, tags, time, yields}.

Storage layout:
- `data/raw/` — downloaded JSON(s) from the source repository.
- `data/processed/recipes.parquet` — cleaned, normalized records.
- `data/processed/chunks.parquet` — retrieval chunks with metadata.

Evaluation split and ground truth:
- Create 50–100 evaluation queries covering constraints, technique, substitutions, and multi-hop scenarios.
- For each query, label 1–3 relevant documents/passages (doc_id + optional chunk indices) as ground truth.
- Hold out this eval set from any prompt or parameter tuning.

Synthetic Q/A generation (for broader testing):
- Use an LLM to generate questions from sampled recipes (prompted to stick to visible content).
- Automatically link generated questions to their source recipe/chunk for weak labels.
- Manually spot-check a subset to ensure label quality.

---

## 3) Architecture Overview

Component mapping to objectives:
- Ingestion → builds knowledge base (clean, chunk, index).
- Retrieval → hybrid search + re-ranking + query rewriting.
- Generation → prompt assembly + LLM with citations.
- Interface → Streamlit UI and FastAPI API.
- Monitoring → feedback logging + dashboard.
- Reproducibility/Containerization → docker-compose + pinned deps.

Components:
- Ingestion:
  - Python script downloads/cleans Open Recipes, normalizes fields, chunks text.
- Index:
  - Vector DB: Qdrant (Docker) with sentence-transformers embeddings.
  - Lexical DB: Meilisearch (Docker) for BM25.
- Retrieval:
  - Hybrid fusion (RRF) of vector and BM25 results.
  - Cross-encoder re-ranking on fused candidates.
  - Query rewriting (HyDE) for underspecified queries.
- Generation (LLM):
  - Pluggable: OpenAI, Groq, or local Ollama.
  - Prompt builder injects top-k chunks + instructions to cite sources.
- API:
  - FastAPI: `/query`, `/feedback`, `/healthz`, `/metrics`.
- UI:
  - Streamlit chat, source citations, feedback controls, settings sidebar.
- Monitoring:
  - Log queries, retrieval diagnostics, latencies, tokens, feedback to DB.
  - Streamlit dashboard with ≥5 charts.
- Containerization:
  - docker-compose: api, ui, qdrant, meilisearch.

Data flow (simplified):
```text
User → UI/API → [Query Rewriter (HyDE)] → Retriever (Vector+BM25) → RRF → Re-ranker →
Prompt Builder (+ top-k chunks with citations) → LLM → Answer + Sources → UI/API
                                     ↘ Logs/Feedback ↙
                                Monitoring Store → Dashboard
```

Configuration toggles (in UI and env):
- Enable/disable: HyDE, re-ranking, hybrid fusion.
- Embedding model and LLM provider/model.
- Max tokens, temperature, and latency/cost guardrails.

---

## 4) Stack and Tools

- Python 3.11
- Retrieval:
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (default), bge-m3 (alternative).
  - Vector: Qdrant (Docker, HTTP API) or Chroma (SQLite fallback for local-only).
  - Lexical: Meilisearch (Docker). Fallback: `rank_bm25` (pure Python) for local dev.
  - Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast); optional larger model for A/B.
  - LLM:
  - OpenAI (gpt-4o-mini / gpt-4.1) default; fallback to Ollama (llama3.1 / mistral) or Groq.
  - Frameworks:
  - FastAPI (API), Streamlit (UI + monitoring), pydantic, uvicorn.
  - Eval:
  - ragas, scikit-learn, numpy/pandas, matplotlib/plotly for plots.
  - Monitoring:
  - sqlite3 + SQLAlchemy; Streamlit dashboard.
  - DevX:
  - pre-commit, black, isort, ruff, mypy (optional).
  - docker, docker-compose.

---

## 5) Retrieval Flow (Detailed)

1) Preprocess:
   - Normalize ingredients list (quantities/units), unify whitespace.
   - Chunking strategy:
     - For instructions: semantic + sliding window (e.g., 500 tokens, 100 overlap).
     - For metadata (title, tags): attach as chunk metadata.

2) Index:
   - Embeddings -> Qdrant collection with payload: {title, url, ingredients, tags, recipe_id, chunk_id, section}.
   - Full doc text to Meilisearch index with fields and searchable attributes.

3) Query pipeline:
   - Optional query rewriting (HyDE) -> combine original + synthetic query.
   - Retrieve:
     - Vector top_k_v (e.g., 20).
     - BM25 top_k_b (e.g., 50).
   - Fuse with RRF -> top N (e.g., 30).
   - Re-rank with CrossEncoder -> top k (e.g., 6).
   - Build prompt with these chunks + citations.
   - Send to LLM with system prompt that enforces grounded answers.

   Parameters and defaults (tunable):
   - top_k_v=20, top_k_b=50, rrf_k=60, final_k=6
   - embed_model=all-MiniLM-L6-v2, cross_encoder=ms-marco-MiniLM-L-6-v2
   - chunk_tokens≈500, chunk_overlap≈100
   - enable_hyde=true, enable_rerank=true, enable_hybrid=true

4) Answer formatting:
   - Markdown with numbered citations like [1], [2].
   - Source list includes title + URL + short snippet.
   - Safety: avoid unsafe substitutions; note uncertainties.

5) Diagnostics (returned in `diagnostics` field):
   - retrieved_ids, scores, rrf_ranks, rerank_scores, latency_ms, token_usage.

---

## 6) Evaluation

- Dataset:
  - 50–100 queries curated across:
    - Ingredient constraints (e.g., “no dairy, <30 minutes”).
    - Technique (“how to temper chocolate”).
    - Substitution (“replace buttermilk?”).
    - Multi-hop (“easy vegan dinner with chickpeas and spinach”).
  - Ground truth: a small set of expected doc IDs/URLs or reference chunks.

- Retrieval metrics:
  - Recall@k (k ∈ {5, 10}), MRR, nDCG. Definitions in `docs/evaluation.md`.
  - Grid search:
    - top_k_v ∈ {10, 20, 40}, top_k_b ∈ {20, 50, 100}, rrf_k ∈ {10, 60}
    - chunk_tokens ∈ {300, 500, 800}, overlap ∈ {50, 100}
    - embed_model ∈ {MiniLM, bge-m3}
    - hyde ∈ {on, off}, rerank ∈ {on, off}
  - Output: CSV of runs with metrics; plots for Recall@k and nDCG by config.

- LLM output:
  - RAGAS: faithfulness, answer relevance, context precision, context recall.
  - LLM-as-judge: rubric on factuality, citation correctness, usefulness.
  - Prompt variants: baseline vs. citation-enforced vs. refusal-focused for low-evidence.

- Report:
  - Save CSV of runs, plots, and summary; pick best config for prod.
  - Fix seed and record versions for reproducibility.

---

## 7) Monitoring

- Log per interaction:
  - query, rewritten_query, retrieval_candidates, selected_contexts, model, latency_ms, token_usage, cost_estimate, user_feedback, session_id, timestamp.
- Dashboard (≥5 charts):
  - Queries/day, avg response latency.
  - Retrieval Recall@k over time (rolling with eval subset).
  - Feedback satisfaction rate (% thumbs up).
  - Token usage and cost per day (if OpenAI).
  - Top failing queries (low recall, low feedback) with drill-down.
- Storage: SQLite (`monitoring.db`) via SQLAlchemy; table schemas in `docs/monitoring.md`.

---

## 8) Interface

- Streamlit:
  - Chat pane, response with collapsible sources.
  - Feedback controls with comment box.
  - Settings sidebar for:
    - LLM provider/model.
    - Enable/disable HyDE, re-ranking, adjust k values.
  - Upload option for personal recipe PDFs/URLs (optional stretch).

- FastAPI:
  - POST `/query`
    - Request:
      ```json
      {"query": "vegan dinner under 30 minutes with chickpeas", "options": {"hyde": true, "rerank": true, "k": 6}}
      ```
    - Response (truncated):
      ```json
      {"answer": "...", "sources": [{"title": "...", "url": "...", "snippet": "..."}], "diagnostics": {"latency_ms": 850}}
      ```
  - POST `/feedback` {"message_id": "...", "rating": "up|down", "comment": "..."}
  - GET `/healthz`
  - GET `/metrics` (basic stats)

---

## 9) Ingestion Pipeline

- Python script with CLI (Typer or argparse):
  - `download_openrecipes --out data/raw/`
  - `normalize_and_chunk --in data/raw/ --out data/processed/ --chunk-tokens 500 --overlap 100`
  - `build_vector_index --chunks data/processed/chunks.parquet --qdrant-url $QDRANT_URL`
  - `build_lexical_index --docs data/processed/recipes.parquet --meili-url $MEILI_URL`
  - `verify_counts --expect-min 100000`
  - Optional: simple dlt/prefect project if you want orchestration.

---

## 10) Repo Structure

```text
receiper/
  app/
    api/
      main.py
      routers/
        query.py
        feedback.py
      core/
        config.py
        logging.py
      rag/
        embedder.py
        hybrid_retriever.py
        reranker.py
        query_rewriter.py
        prompt.py
        pipeline.py
        stores/
          qdrant_store.py
          meili_store.py
      ui/
        streamlit_app.py
        dashboard.py
    data/
      raw/
      processed/
      eval/
        queries.jsonl
        ground_truth.jsonl
    ingestion/
      ingest_openrecipes.py
      chunking.py
      schema.py
    eval/
      retrieval_eval.py
      llm_eval_ragas.py
      report.ipynb
    docker/
      docker-compose.yml
      Dockerfile.api
      Dockerfile.ui
    scripts/
      quickstart.sh
      create_eval_set.py
    tests/
      test_retrieval.py
      test_pipeline.py
    .env.example
    requirements.txt
    Makefile
    README.md
    docs/
      setup.md
      usage.md
      evaluation.md
      monitoring.md
      architecture.md
```

   Notes:
   - `tests/` will include small fixtures to unit-test retrieval and pipeline glue.
   - `scripts/quickstart.sh` runs end-to-end: create env, download data, ingest, start services.

---

## 11) Configuration

- `.env.example`:
  - OPENAI_API_KEY=
  - QDRANT_URL=http://qdrant:6333
  - MEILI_URL=http://meilisearch:7700
  - MEILI_MASTER_KEY=changeme
  - MODEL_EMBED=all-MiniLM-L6-v2
  - MODEL_CROSS_ENCODER=ms-marco-MiniLM-L-6-v2
  - LLM_PROVIDER=openai|ollama|groq
  - OPENAI_MODEL=gpt-4o-mini
  - OLLAMA_MODEL=llama3.1
  - API_HOST=0.0.0.0
  - API_PORT=8000
  - UI_PORT=8501
  - LOG_LEVEL=INFO
- `app/api/core/config.py` loads env with pydantic.
- `docs/setup.md` provides step-by-step environment setup.

---

## 12) Containerization

- docker-compose services:
  - api (FastAPI), ui (Streamlit), qdrant, meilisearch.
- Volumes for Qdrant and Meili persistence.
- Healthchecks.
- Make targets:
  - make up, make down, make ingest, make eval, make ui, make api.

   docker/docker-compose.yml outline:
   - qdrant: use `qdrant/qdrant:latest`, expose 6333, volume `qdrant_data`.
   - meilisearch: `getmeili/meilisearch:latest`, env `MEILI_MASTER_KEY`, volume `meili_data`.
   - api: build from `Dockerfile.api`, depends_on qdrant/meilisearch with healthcheck, exposes 8000.
   - ui: build from `Dockerfile.ui`, depends_on api, exposes 8501.

   Healthchecks:
   - qdrant: GET http://qdrant:6333/ready
   - meilisearch: GET http://meilisearch:7700/health
   - api: GET http://api:8000/healthz

---

## 13) Documentation Plan

- README includes:
  - Problem description, features.
  - Architecture diagram and flowchart.
  - Setup (local and docker), quickstart.
  - Evaluation results with screenshots.
  - Monitoring dashboard screenshots.
  - UI GIF/video.
  - Mapping to evaluation criteria checklist.
- Sub-files:
  - docs/setup.md, usage.md, evaluation.md, monitoring.md, architecture.md.
- Add a 30–60s demo video.
- Peer review helper: include repo URL + a sample commit hash and instructions for `git reset --hard`.
- Version pinning: include exact versions in `requirements.txt` and model names in README.

---

## 14) Timeline (7–10 days)

- Day 1: Repo scaffold, ingestion script, initial index.
- Day 2: Qdrant + Meili in compose, hybrid retrieval.
- Day 3: Re-ranking, HyDE.
- Day 4: FastAPI endpoints.
- Day 5: Streamlit UI.
- Day 6: Eval set + retrieval evaluation.
- Day 7: LLM output eval + tuning.
- Day 8: Monitoring + dashboard.
- Day 9: Docker polish, README + docs + screenshots + video.
- Day 10: Optional cloud deploy (Railway/Render).

   Risk buffer: keep 10–20% time for data quirks and performance tuning.

---

## 15) Risks and Mitigations

- Data quality variance:
  - Normalize aggressively; fallback to title/ingredients for poor instructions.
- LLM cost:
  - Default to fast/cheap models; enable Ollama.
- Latency:
  - Use MiniLM embeddings and mini cross-encoder; cache results.
- Docker/DNS flakiness:
  - Add healthchecks and retries; document local ports and env.

---

## 16) Success Criteria (mapped to rubric)

- Retrieval flow: Vector + LLM (2/2).
- Retrieval evaluation: Multiple approaches, pick best (2/2).
- LLM evaluation: Multiple prompts/configs (2/2).
- Interface: Streamlit UI (2/2).
- Ingestion pipeline: Automated scripts (2/2).
- Monitoring: Feedback + 5+ charts (2/2).
- Containerization: Full docker-compose (2/2).
- Reproducibility: Clear instructions + versions (2/2).
- Best practices: Hybrid, re-ranking, query rewriting (+3).
- Bonus: Cloud deployment (+2).

   Checklist in README will explicitly map features to each criterion for easy review.

---

## 17) Concrete Milestones and Deliverables

- Ingestion produces:
  - `data/processed/chunks.parquet`
- Indexing:
  - Qdrant collection `recipes_chunks`
  - Meili index `recipes_docs`
- API:
  - `POST /query` returns answer, citations, diagnostics
  - `POST /feedback` persists
- UI:
  - `streamlit run app/ui/streamlit_app.py`
- Monitoring:
  - `streamlit run app/ui/dashboard.py`
- Evaluation:
  - `python eval/retrieval_eval.py --grid default`
  - `python eval/llm_eval_ragas.py`
- Containerization:
  - `docker compose up -d` brings up all services; healthchecks pass.
- Reproducibility:
  - `Makefile` targets: setup, ingest, up, down, eval, ui, api, clean.

---

## 18) Next Decisions

- Default LLM provider (OpenAI vs Ollama).
- BM25 backend: Meilisearch (Docker) vs Python fallback for simplicity.
- Cloud deploy target (Render/Railway/Fly.io) if pursuing bonus points.
