#!/usr/bin/env python3
"""Run one system over one evaluation split and save every record to disk.

Systems:
    no_retrieval : the LLM answers from its own knowledge (same model as RAG)
    dense_rag    : top-k dense retrieval, then the LLM answers from the sources

Each output line holds everything needed to recompute scores WITHOUT new LLM
calls: question, retrieved chunks with their text, the prompt-visible source
tags, the answer, model version, usage and the config used.

The run is resumable (finished question ids are skipped) and a successful LLM
call is never repeated (the response cache is keyed by prompt).

Test-set guard: running on the test split needs --confirm-frozen, a conscious
statement that code, prompts and configs are final (see RESEARCH_LOG.md).

Examples:
    python src/run.py --split dev --system dense_rag --no-generate     # retrieval only, no LLM calls
    python src/run.py --split dev --system dense_rag
    python src/run.py --split dev --system no_retrieval
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

from generation import (INSUFFICIENT, SYSTEM_NO_RETRIEVAL, SYSTEM_RAG, LLMClient,
                        LLMError, QuotaExceeded, load_config, no_retrieval_prompt,
                        rag_prompt)

PRODUCTS = os.path.join("data", "processed", "products.jsonl")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build_sources(hits, products):
    """Turn retrieved chunks into prompt sources S1..Sk (title shown on each)."""
    sources = []
    for h in hits:
        lines = h["text"].split("\n")
        body = "\n".join(lines[1:] if h["source_type"] == "metadata" else lines[2:])
        sources.append({
            "tag": f"S{h['rank']}",
            "type": h["source_type"],
            "product_title": products[h["parent_asin"]]["title"],
            "rating": h["rating"],
            "text": body,
        })
    return sources


def cited_tags(answer):
    tags = []
    for group in re.findall(r"\[([^\]]*)\]", answer or ""):
        tags += [f"S{n}" for n in re.findall(r"S(\d+)", group)]
    return sorted(set(tags), key=lambda t: int(t[1:]))


def is_abstention(answer):
    return (answer or "").lstrip(" \t\n*_\"'").upper().startswith(INSUFFICIENT)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--system", required=True, choices=["no_retrieval", "dense_rag"])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--no-generate", action="store_true",
                        help="retrieval only: no LLM calls (dense_rag only)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--confirm-frozen", action="store_true",
                        help="required for --split test")
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_frozen:
        raise SystemExit("[ERROR] The test split is reserved for the final evaluation. "
                         "Pass --confirm-frozen only after code, prompts and configs are final.")
    if args.no_generate and args.system != "dense_rag":
        raise SystemExit("[ERROR] --no-generate only makes sense for dense_rag.")

    cfg = load_config(args.config)
    k = cfg["retrieval"]["top_k"]
    questions = list(read_jsonl(os.path.join("evaluation", f"{args.split}.jsonl")))
    products = {p["parent_asin"]: p for p in read_jsonl(PRODUCTS)}

    out_dir = args.out_dir or os.path.join("outputs", args.split)
    os.makedirs(out_dir, exist_ok=True)
    name = args.system + ("_retrieval_only" if args.no_generate else "")
    out_path = os.path.join(out_dir, f"{name}.jsonl")
    done = {r["id"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()

    retriever = None
    if args.system == "dense_rag":
        from retrieval import Retriever
        retriever = Retriever(cfg)
    client = None if args.no_generate else LLMClient(cfg)

    print(f"{name}: {len(questions)} questions, {len(done)} already done -> {out_path}")
    failed = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for q in questions:
            if q["id"] in done:
                continue
            record = {"id": q["id"], "split": args.split, "category": q["category"],
                      "system": args.system, "question": q["question"],
                      "config": args.config, "top_k": k if retriever else 0,
                      "retrieval_mode": cfg["retrieval"].get("mode", "dense") if retriever else None,
                      "retrieved": [], "sources_shown": [], "answer": None,
                      "abstained": None, "cited_tags": [], "model_version": None,
                      "finish_reason": None, "usage": None, "cache_hit": None,
                      "error": None,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            try:
                if retriever:
                    record["retrieved"] = retriever.retrieve(q["question"], cfg, q["product_asins"])
                    sources = build_sources(record["retrieved"], products)
                    record["sources_shown"] = [s["tag"] for s in sources]
                if client:
                    if retriever:
                        res = client.generate(SYSTEM_RAG, rag_prompt(q["question"], sources),
                                              tag=f"{args.split}/{name}/{q['id']}")
                    else:
                        res = client.generate(SYSTEM_NO_RETRIEVAL,
                                              no_retrieval_prompt(q["question"]),
                                              tag=f"{args.split}/{name}/{q['id']}")
                    record.update(answer=res["text"], abstained=is_abstention(res["text"]),
                                  cited_tags=cited_tags(res["text"]),
                                  model_version=res["model_version"],
                                  finish_reason=res["finish_reason"], usage=res["usage"],
                                  cache_hit=res["cache_hit"])
            except QuotaExceeded as exc:
                print(f"[STOP] {exc}")
                break
            except LLMError as exc:
                failed += 1
                record["error"] = str(exc)
                print(f"[ERROR] {q['id']}: {exc}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {q['id']} done" + (" (error)" if record["error"] else ""))
    if client:
        print(f"real LLM calls this run: {client.calls_made}; cache hits: {client.cache_hits}; "
              f"failed questions: {failed}")


if __name__ == "__main__":
    main()
