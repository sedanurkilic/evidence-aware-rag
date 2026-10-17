#!/usr/bin/env python3
"""Dense retrieval over the ChromaDB index (baseline).

The retrieval unit is a chunk; every hit carries the source document id, the
source type (review / metadata) and the product id, so metrics can be computed
at document level from saved outputs.

Usage (quick manual check, no LLM call):
    python src/retrieval.py "Does this product contain latex?"
"""

import argparse

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


def load_config(path="configs/baseline.yaml"):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Retriever:
    def __init__(self, cfg):
        self.model = SentenceTransformer(cfg["embedding"]["model"], device="cpu")
        self.model.max_seq_length = cfg["embedding"]["max_seq_length"]
        client = chromadb.PersistentClient(path=cfg["index"]["chroma_path"])
        self.collection = client.get_collection(cfg["index"]["collection"])

    def embed(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def search(self, query, k, source_type=None, exclude_asins=None):
        """Top-k chunks by cosine similarity; optional filter on source type.
        exclude_asins is DIAGNOSTIC ONLY (uses gold labels; Experiment 2)."""
        conds = []
        if source_type:
            conds.append({"source_type": source_type})
        if exclude_asins:
            conds.append({"parent_asin": {"$nin": list(exclude_asins)}})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        res = self.collection.query(
            query_embeddings=[self.embed(query)], n_results=k, where=where,
            include=["documents", "metadatas", "distances"])
        hits = []
        for rank, (chunk_id, text, meta, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                    res["distances"][0]), start=1):
            hits.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "source_type": meta["source_type"],
                "parent_asin": meta["parent_asin"],
                "rating": meta["rating"] if meta["source_type"] == "review" else None,
                "score": round(1.0 - dist, 6),     # cosine similarity
                "text": text,
            })
        return hits


def _retrieve(self, query, cfg, exclude_asins=None):
    """Retrieval as configured. mode 'dense' (baseline): plain top-k.
    mode 'source_aware' (Experiment 1): review_k review chunks + metadata_k metadata
    chunks, each the best by cosine similarity within its source type; merged by score.
    The total number of chunks is unchanged (review_k + metadata_k == top_k)."""
    rcfg = cfg["retrieval"]
    if rcfg.get("mode", "dense") == "dense":
        return self.search(query, rcfg["top_k"])
    assert rcfg["review_k"] + rcfg["metadata_k"] == rcfg["top_k"], "context budget must stay fixed"
    hits = (self.search(query, rcfg["review_k"], "review")
            + self.search(query, rcfg["metadata_k"], "metadata",
                          exclude_asins if rcfg.get("diagnostic_exclude_gold_metadata") else None))
    hits.sort(key=lambda h: -h["score"])
    for rank, h in enumerate(hits, start=1):
        h["rank"] = rank
    return hits


Retriever.retrieve = _retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("-k", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    k = args.k or cfg["retrieval"]["top_k"]
    for h in Retriever(cfg).search(args.query, k):
        preview = h["text"].replace("\n", " | ")[:140]
        print(f"{h['rank']}. {h['source_type']:<8} {h['score']:.3f} {h['doc_id']}  {preview}")


if __name__ == "__main__":
    main()
