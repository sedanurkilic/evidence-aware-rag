#!/usr/bin/env python3
"""Build the ChromaDB vector index over the fixed corpus.

Two source types are indexed in ONE collection so a query can retrieve both:
  * metadata : one document per product (title, store, price, categories,
               rating summary, features, description, details)
  * review   : one document per kept review

Design decisions (also recorded in RESEARCH_LOG.md):
  * Fair baseline: every chunk starts with the product title. Review text rarely
    names the product, so without it a question that names the product could
    never match its reviews and the baseline would be a straw man.
  * Chunks are packed to at most `embedding.max_seq_length` model tokens
    (including the header) so nothing is silently truncated; the source
    document id is kept on every chunk.
  * Embeddings are L2-normalized, cosine distance, CPU only.

Usage (repository root):
    python src/index.py
"""

import argparse
import json
import os
import statistics
import time
from collections import Counter

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

PROC = os.path.join("data", "processed")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def make_counter(tokenizer):
    def ntok(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    return ntok


def fit_words(header, unit, ntok, max_tokens):
    """Split one over-long unit into word-based pieces that fit with the header."""
    pieces, buf = [], []
    for word in unit.split():
        trial = " ".join(buf + [word])
        if buf and ntok(header + "\n" + trial) > max_tokens:
            pieces.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def pack(header, units, ntok, max_tokens):
    """Greedy-pack units after the header into chunks of <= max_tokens tokens."""
    chunks, current = [], []

    def text_of(parts):
        return header + "\n" + "\n".join(parts)

    for unit in units:
        pieces = [unit]
        if ntok(text_of([unit])) > max_tokens:
            pieces = fit_words(header, unit, ntok, max_tokens)
        for piece in pieces:
            if current and ntok(text_of(current + [piece])) > max_tokens:
                chunks.append(text_of(current))
                current = []
            current.append(piece)
    if current:
        chunks.append(text_of(current))
    return chunks or [header]


def metadata_units(p):
    units = []
    if p.get("store"):
        units.append(f"Brand/store: {p['store']}")
    if p.get("price") not in (None, ""):
        units.append(f"Price: {p['price']}")
    cats = p.get("categories") or []
    if cats:
        units.append("Categories: " + " > ".join(str(c) for c in cats))
    if p.get("average_rating") is not None:
        units.append(f"Average rating: {p['average_rating']} from {p.get('rating_number')} ratings")
    units += [f"Feature: {f}" for f in p["features"]]
    units += [f"Description: {d}" for d in p["description"]]
    for key, value in (p.get("details") or {}).items():
        units.append(f"{key}: {value}")
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    max_tokens = cfg["embedding"]["max_seq_length"]
    icfg = cfg["index"]

    started = time.time()
    model = SentenceTransformer(cfg["embedding"]["model"], device="cpu")
    model.max_seq_length = max_tokens
    ntok = make_counter(model.tokenizer)

    products = {p["parent_asin"]: p for p in read_jsonl(os.path.join(PROC, "products.jsonl"))}
    items = []   # (chunk_id, text, metadata)

    for p in products.values():
        header = f"Product metadata: {p['title']}"
        chunks = pack(header, metadata_units(p), ntok, max_tokens)
        doc_id = f"meta_{p['parent_asin']}"
        for i, text in enumerate(chunks):
            items.append((doc_id if len(chunks) == 1 else f"{doc_id}#c{i}", text, {
                "doc_id": doc_id, "source_type": "metadata", "parent_asin": p["parent_asin"],
                "chunk_index": i, "n_chunks": len(chunks), "rating": -1.0}))

    n_reviews = 0
    for r in read_jsonl(os.path.join(PROC, "reviews.jsonl")):
        n_reviews += 1
        p = products[r["parent_asin"]]
        header = f"{p['title']}\n{r['rating']:.0f}-star customer review"
        body = f"{r['title']}. {r['text']}" if r["title"] else r["text"]
        chunks = pack(header, [body], ntok, max_tokens)
        for i, text in enumerate(chunks):
            items.append((r["doc_id"] if len(chunks) == 1 else f"{r['doc_id']}#c{i}", text, {
                "doc_id": r["doc_id"], "source_type": "review", "parent_asin": r["parent_asin"],
                "chunk_index": i, "n_chunks": len(chunks), "rating": float(r["rating"])}))

    print(f"chunks to embed: {len(items)} (products {len(products)}, reviews {n_reviews})")

    client = chromadb.PersistentClient(path=icfg["chroma_path"])
    try:
        client.delete_collection(icfg["collection"])
    except Exception:  # noqa: BLE001 - collection did not exist yet
        pass
    collection = client.create_collection(icfg["collection"],
                                          metadata={"hnsw:space": "cosine"})

    batch = icfg["batch_size"]
    for start in range(0, len(items), batch):
        part = items[start:start + batch]
        embeddings = model.encode([t for _, t, _ in part], batch_size=batch,
                                  normalize_embeddings=True, show_progress_bar=False)
        collection.add(ids=[i for i, _, _ in part],
                       embeddings=embeddings.tolist(),
                       documents=[t for _, t, _ in part],
                       metadatas=[m for _, _, m in part])
        if (start // batch) % 20 == 0:
            print(f"  embedded {min(start + batch, len(items))}/{len(items)}")

    lengths = [ntok(t) for _, t, _ in items]
    by_type = Counter(m["source_type"] for _, _, m in items)
    stats = {
        "embedding_model": cfg["embedding"]["model"],
        "max_seq_length": max_tokens,
        "chunks_total": len(items),
        "chunks_by_source_type": dict(by_type),
        "multi_chunk_documents": len({m["doc_id"] for _, _, m in items if m["n_chunks"] > 1}),
        "chunk_tokens": {"median": statistics.median(lengths), "max": max(lengths),
                         "over_limit": sum(1 for n in lengths if n > max_tokens)},
        "collection_count": collection.count(),
        "seconds": round(time.time() - started, 1),
    }
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "index_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
