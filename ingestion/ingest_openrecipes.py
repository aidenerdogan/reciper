import os
import json
import pathlib
import typer
from typing import Optional
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import meilisearch

from ingestion.schema import Recipe, Chunk
from ingestion.chunking import chunk_text

app = typer.Typer(help="Open Recipes ingestion CLI")

@app.command()
def download(out: pathlib.Path = typer.Option(..., help="Output folder")):
    out.mkdir(parents=True, exist_ok=True)
    # Placeholder: instruct user to download manually; automate later
    (out / "README.txt").write_text("Place Open Recipes JSON files here. Automation TBD.")
    typer.echo(f"Prepared folder: {out}")

@app.command()
def process(
    inp: pathlib.Path = typer.Option(..., help="Raw data folder"),
    out: pathlib.Path = typer.Option(..., help="Processed folder"),
    chunk_tokens: int = 500,
    overlap: int = 100,
):
    out.mkdir(parents=True, exist_ok=True)

    def load_json_files(folder: pathlib.Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        paths = list(folder.rglob("*.json")) + list(folder.rglob("*.jsonl")) + list(folder.rglob("*.json.gz"))
        for p in paths:
            try:
                # Prefer streaming to handle large files
                if p.suffix == ".gz":
                    import gzip
                    fh = gzip.open(p, "rt", encoding="utf-8")
                else:
                    fh = p.open("r", encoding="utf-8")
                with fh as f:
                    content = f.read()
                # First try standard JSON (array or object)
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        maybe_list = data.get("recipes") if "recipes" in data else None
                        if isinstance(maybe_list, list):
                            records.extend(maybe_list)
                        else:
                            records.append(data)
                    else:
                        raise ValueError("Unsupported JSON root type")
                except Exception:
                    # Fallback: JSON Lines (one JSON object per line)
                    count_before = len(records)
                    for line in content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                records.append(obj)
                            elif isinstance(obj, list):
                                records.extend(obj)
                        except Exception:
                            # Skip malformed lines but continue
                            continue
                    parsed = len(records) - count_before
                    if parsed == 0:
                        raise ValueError("No records parsed from JSON or JSON Lines")
            except Exception as e:
                typer.echo(f"Failed to parse {p}: {e}")
        return records

    raw = load_json_files(inp)
    if not raw:
        typer.echo("No JSON files found in input folder.")
    
    recipes: List[Recipe] = []
    for r in raw:
        try:
            rid = str(r.get("id") or r.get("_id") or r.get("url") or r.get("title") or len(recipes))
            title = (r.get("title") or "").strip()
            url = r.get("url")
            ingredients = r.get("ingredients") or r.get("normalized_ingredients") or []
            if isinstance(ingredients, str):
                ingredients = [ingredients]
            instructions = r.get("instructions") or r.get("directions") or []
            tags = r.get("tags") or r.get("categories") or []
            time = r.get("time") or {k: r.get(k) for k in ("prep", "cook", "total") if r.get(k) is not None}
            yields = r.get("yields") or r.get("servings")

            recipe = Recipe(
                id=rid,
                title=title,
                url=url,
                ingredients=list(ingredients),
                instructions=instructions,
                tags=list(tags) if tags else None,
                time=time or None,
                yields=str(yields) if yields is not None else None,
            )
            recipes.append(recipe)
        except Exception:
            continue

    # Build DataFrame for recipes
    rec_df = pd.DataFrame([r.model_dump() for r in recipes])
    rec_out = out / "recipes.parquet"
    rec_df.to_parquet(rec_out, index=False)

    # Create chunks: ingredients as one chunk, plus instruction chunks
    chunk_rows: List[Dict[str, Any]] = []
    for r in recipes:
        base_meta = {
            "doc_id": r.id,
            "title": r.title,
            "url": r.url,
            "tags": r.tags,
            "time": r.time,
            "yields": r.yields,
        }
        # Ingredients chunk
        if r.ingredients:
            chunk_rows.append(
                {
                    **base_meta,
                    "section": "ingredients",
                    "position": 0,
                    "text": "\n".join(r.ingredients),
                }
            )
        # Instruction chunks
        instr_text = r.instructions if isinstance(r.instructions, str) else "\n".join(r.instructions)
        for i, ch in enumerate(chunk_text(instr_text, max_tokens=chunk_tokens, overlap=overlap)):
            chunk_rows.append(
                {
                    **base_meta,
                    "section": "instructions",
                    "position": i,
                    "text": ch,
                }
            )

    chunks_df = pd.DataFrame(chunk_rows)
    chunks_out = out / "chunks.parquet"
    chunks_df.to_parquet(chunks_out, index=False)

    typer.echo(f"Wrote {len(rec_df)} recipes -> {rec_out}")
    typer.echo(f"Wrote {len(chunks_df)} chunks -> {chunks_out}")

@app.command()
def index(
    chunks: pathlib.Path = typer.Option(..., help="Chunks parquet path"),
    collection: str = typer.Option("recipes_chunks", help="Qdrant collection"),
    meili_index: str = typer.Option("recipes_chunks", help="Meilisearch index"),
):
    # Load data
    df = pd.read_parquet(chunks)
    if df.empty:
        typer.echo("No chunks to index.")
        raise typer.Exit(1)

    # Clients
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    meili_url = os.getenv("MEILI_URL", "http://localhost:7700")
    meili_key = os.getenv("MEILI_MASTER_KEY", "changeme")
    model_name = os.getenv("MODEL_EMBED", "all-MiniLM-L6-v2")

    typer.echo(f"Connecting to Qdrant: {qdrant_url}")
    qc = QdrantClient(url=qdrant_url)
    embedder = SentenceTransformer(model_name)
    dim = embedder.get_sentence_embedding_dimension()

    # Ensure collection
    collections = {c.name for c in qc.get_collections().collections}
    if collection not in collections:
        qc.create_collection(collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    # Helpers
    def sanitize(obj):
        try:
            import numpy as np
        except Exception:
            np = None  # type: ignore

        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        if np is not None:
            if isinstance(obj, np.ndarray):
                return [sanitize(v) for v in obj.tolist()]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
        return str(obj)

    # Prepare points and upsert in batches to avoid timeouts
    texts = df["text"].astype(str).tolist()
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    records = df.to_dict(orient="records")
    total = len(records)
    batch = 500
    typer.echo(f"Upserting {total} points into Qdrant collection '{collection}' in batches of {batch} ...")
    cur = 0
    while cur < total:
        end = min(cur + batch, total)
        pts = []
        for idx in range(cur, end):
            row = records[idx]
            vec = vectors[idx]
            payload = {k: row[k] for k in row.keys() if k != "text"}
            payload["text"] = row["text"]
            payload = sanitize(payload)
            pts.append(PointStruct(id=idx, vector=vec, payload=payload))
        try:
            qc.upsert(collection_name=collection, points=pts)
        except Exception as e:
            typer.echo(f"Upsert failed for batch {cur}-{end}: {e}. Retrying once...")
            qc.upsert(collection_name=collection, points=pts)
        cur = end

    # Meilisearch indexing (BM25)
    typer.echo(f"Connecting to Meilisearch: {meili_url}")
    ms = meilisearch.Client(meili_url, meili_key)
    try:
        ms.index(meili_index).get_raw_info()
    except Exception:
        ms.create_index(uid=meili_index, options={"primaryKey": "_id"})

    # Build chunk docs
    meili_docs = []
    for i, row in enumerate(df.to_dict(orient="records")):
        doc = {
            "_id": f"{row['doc_id']}::{row['section']}::{row['position']}",
            "doc_id": row["doc_id"],
            "title": row.get("title"),
            "url": row.get("url"),
            "section": row.get("section"),
            "position": row.get("position"),
            "text": row.get("text"),
            "tags": row.get("tags"),
        }
        meili_docs.append(sanitize(doc))
    typer.echo(f"Indexing {len(meili_docs)} docs into Meilisearch index '{meili_index}' ...")
    ms.index(meili_index).add_documents(meili_docs)
    typer.echo("Indexing complete.")

if __name__ == "__main__":
    app()
