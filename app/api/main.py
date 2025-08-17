import os
from typing import Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter as QFilter
import meilisearch

APP_VERSION = "0.1.0"

app = FastAPI(title="Receiper API", version=APP_VERSION)


class QueryRequest(BaseModel):
    query: str
    options: dict | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    rating: str
    comment: str | None = None


# --- Clients and models ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "changeme")
EMBED_MODEL = os.getenv("MODEL_EMBED", "all-MiniLM-L6-v2")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes_chunks")
MEILI_INDEX = os.getenv("MEILI_INDEX", "recipes_chunks")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Re-ranking config
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))

_embedder: SentenceTransformer | None = None
_qdrant: QdrantClient | None = None
_meili: meilisearch.Client | None = None
_reranker: CrossEncoder | None = None


def get_clients():
    global _embedder, _qdrant, _meili
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    if _meili is None:
        _meili = meilisearch.Client(MEILI_URL, MEILI_MASTER_KEY)
    return _embedder, _qdrant, _meili


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


@app.get("/healthz")
async def healthz():
    try:
        _, q, m = get_clients()
        # lightweight checks
        q.get_collections()
        m.health()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


def rrf_fuse(results: Dict[str, Dict[str, float]], k: int = 60) -> List[tuple]:
    # results: source_name -> {id -> rank (1-based) or score}
    # We'll treat values as rank positions; if they are scores, convert to ranks by ordering.
    # Implement reciprocal rank fusion: sum(1/(k + rank))
    # Normalize input if provided as scores (higher is better)
    fused: Dict[str, float] = {}
    for src, mapping in results.items():
        # If values look like floats > 1, treat as scores -> assign ranks
        items = list(mapping.items())
        # Sort by descending value (higher better) to get rank positions
        items.sort(key=lambda x: x[1], reverse=True)
        for rank_pos, (doc_id, _) in enumerate(items, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank_pos)
    # Sort fused
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def compose_answer(query: str, sources: List[Dict[str, Any]], max_chars: int = 600) -> str:
    """Compose a concise answer by lightly summarizing the top sources.
    Fallback is to stitch key snippets together.
    """
    if not sources:
        return "I couldn't find relevant recipes. Try rephrasing or broadening the query."
    # Take the first few sources and extract short snippets
    lines: List[str] = []
    taken = 0
    for s in sources:
        title = s.get("title") or "Recipe"
        snippet = (s.get("snippet") or "").strip().replace("\n", " ")
        if not snippet:
            continue
        lines.append(f"- {title}: {snippet}")
        taken += 1
        if taken >= 3:
            break
    if not lines:
        return "Top results retrieved; open Sources for details."
    intro = "Here are ideas based on top matching recipes:"
    answer = intro + "\n" + "\n".join(lines)
    return answer[:max_chars]


def generate_answer_with_openai(query: str, sources: List[Dict[str, Any]], max_chars: int = 800) -> str:
    """Generate an answer using OpenAI Chat Completions if API key is available."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Build context from sources
        ctx_lines = []
        for i, s in enumerate(sources[:6], 1):
            title = s.get("title") or "Source"
            url = s.get("url") or ""
            snippet = s.get("snippet") or ""
            ctx_lines.append(f"[{i}] {title} {url}\n{snippet}")
        context = "\n\n".join(ctx_lines) if ctx_lines else ""
        system = (
            "You are a helpful cooking assistant. Answer concisely (bullets ok), "
            "grounded strictly in the provided recipe snippets. Include practical steps and key measurements when relevant."
        )
        user = (
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Write a short helpful answer using the sources. If uncertain, say so."
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt[:max_chars] if txt else ""
    except Exception as e:
        raise RuntimeError(f"OpenAI generation failed: {e}")


@app.post("/query")
async def query(req: QueryRequest):
    embedder, qdrant, meili = get_clients()
    query_text = req.query.strip()
    opts = req.options or {}
    top_k = int(opts.get("k", 6))
    use_llm = bool(opts.get("llm", True))
    use_rerank = bool(opts.get("rerank", RERANK_ENABLED))
    rerank_candidates = int(opts.get("rerank_candidates", RERANK_CANDIDATES))

    diagnostics: Dict[str, Any] = {"top_k": top_k}

    # Vector search (Qdrant)
    vec = embedder.encode([query_text], normalize_embeddings=True)[0]
    q_hits = []
    try:
        qres = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=max(20, top_k))
        for h in qres:
            pid = str(h.id)
            payload = h.payload or {}
            q_hits.append({"id": pid, "score": float(h.score), "payload": payload})
    except Exception as e:
        diagnostics["qdrant_error"] = str(e)

    # BM25 search (Meilisearch)
    m_hits = []
    try:
        mres = meili.index(MEILI_INDEX).search(query_text, {"limit": max(20, top_k)})
        for i, hit in enumerate(mres.get("hits", []), start=1):
            m_hits.append({"id": hit.get("_id", str(i)), "score": float(hit.get("_matchesPosition", {}).get("text", [{}])[0].get("start", 0) if "_matchesPosition" in hit else 1.0/(i+1)), "payload": hit})
    except Exception as e:
        diagnostics["meili_error"] = str(e)

    # Prepare maps for RRF using scores as sorting signals
    q_map = {h["id"]: h["score"] for h in q_hits}
    m_map = {h["id"]: h["score"] for h in m_hits}
    fused = rrf_fuse({"qdrant": q_map, "meili": m_map})

    # Build sources list using payloads; prefer qdrant payload if available
    payload_lookup: Dict[str, Dict[str, Any]] = {h["id"]: h["payload"] for h in q_hits}
    for h in m_hits:
        payload_lookup.setdefault(h["id"], h["payload"])

    # Optional re-ranking using a cross-encoder on top fused candidates
    reranked_ids: List[str] | None = None
    if use_rerank:
        try:
            reranker = get_reranker()
            # Build candidate pairs (query, text)
            candidates = fused[: max(top_k, rerank_candidates)]
            pairs = []
            cand_ids = []
            for doc_id, _ in candidates:
                p = payload_lookup.get(doc_id, {})
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                pairs.append((query_text, text))
                cand_ids.append(doc_id)
            if pairs:
                scores = reranker.predict(pairs)
                # sort by score desc
                scored = list(zip(cand_ids, [float(s) for s in scores]))
                scored.sort(key=lambda x: x[1], reverse=True)
                reranked_ids = [doc_id for doc_id, _ in scored]
                diagnostics["rerank_model"] = RERANK_MODEL
                diagnostics["rerank_enabled"] = True
                diagnostics["rerank_candidates_used"] = len(reranked_ids)
        except Exception as e:
            diagnostics["rerank_error"] = str(e)
            diagnostics["rerank_enabled"] = False

    sources = []
    import re, urllib.parse
    ws_re = re.compile(r"\s+")
    ordering = reranked_ids if reranked_ids else [doc_id for doc_id, _ in fused]
    for doc_id in ordering[:top_k]:
        # Find original score from fused for reference
        fused_score = next((s for i, s in fused if i == doc_id), 0.0)
        p = payload_lookup.get(doc_id, {})
        url = p.get("url")
        title = p.get("title") or p.get("name")
        if not title and url:
            try:
                host = urllib.parse.urlparse(url).netloc
                title = host or "Source"
            except Exception:
                title = "Source"
        raw = (p.get("text") or "").strip()
        snippet = ws_re.sub(" ", raw)[:300]
        sources.append(
            {
                "id": doc_id,
                "title": title,
                "url": url,
                "section": p.get("section"),
                "position": p.get("position"),
                "snippet": snippet,
                "score": fused_score,
            }
        )

    # Answer generation
    answer = None
    if use_llm:
        try:
            answer = generate_answer_with_openai(query_text, sources)
            diagnostics["llm_provider"] = LLM_PROVIDER
            diagnostics["llm_model"] = OPENAI_MODEL
        except Exception as e:
            diagnostics["llm_error"] = str(e)
    if not answer:
        answer = compose_answer(query_text, sources)

    return {
        "answer": answer,
        "sources": sources,
        "diagnostics": {
            **diagnostics,
            "qdrant_hits": len(q_hits),
            "meili_hits": len(m_hits),
        },
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    # Placeholder feedback handler
    return {"status": "received"}
