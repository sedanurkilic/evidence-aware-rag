#!/usr/bin/env python3
"""Build the fixed corpus used by every system and every experiment.

Reads the raw Amazon Reviews 2023 (Health & Personal Care) files, applies the
sampling and cleaning rules from configs/baseline.yaml (section `corpus`) and
writes:
    data/processed/products.jsonl   one record per sampled product (metadata)
    data/processed/reviews.jsonl    one record per kept review
    data/processed/corpus_stats.json  counts, parameters, checksums

Rules (all deterministic, seed from the config):
  1. candidate products: at least `min_reviews_per_product` reviews AND usable
     metadata (title + some feature/description/details text);
  2. candidates are shuffled with the seed; products are taken in that order
     until both `target_products` and `min_total_reviews` are reached;
  3. per product: drop empty reviews, drop exact duplicates (same normalized
     text within the product, earliest kept), then keep at most
     `max_reviews_per_product` (seeded sample);
  4. each product gets a `split_pool` (dev / test) so evaluation questions can
     be written from disjoint products.
Source identifiers (asin, parent_asin, user_id, timestamp) are kept.

Usage (repository root):
    python src/data.py
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

import yaml

RAW_REVIEWS = os.path.join("data", "raw", "Health_and_Personal_Care.jsonl.gz")
RAW_META = os.path.join("data", "raw", "meta_Health_and_Personal_Care.jsonl.gz")
OUT_DIR = os.path.join("data", "processed")


def stream(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize(text):
    return " ".join(text.lower().split())


def str_list(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def has_usable_metadata(rec):
    if not (rec.get("title") or "").strip():
        return False
    details = rec.get("details")
    return bool(str_list(rec.get("features")) or str_list(rec.get("description"))
                or (isinstance(details, dict) and details))


def review_doc_id(rec, text, taken):
    digest = hashlib.md5(
        f"{rec.get('user_id')}|{rec.get('timestamp')}|{text}".encode("utf-8")
    ).hexdigest()[:8]
    base = f"rev_{rec['parent_asin']}_{digest}"
    doc_id, n = base, 1
    while doc_id in taken:
        n += 1
        doc_id = f"{base}_{n}"
    taken.add(doc_id)
    return doc_id


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed = cfg["seed"]
    c = cfg["corpus"]

    print("pass 1/4: metadata (usable flags)")
    usable = {rec["parent_asin"] for rec in stream(RAW_META) if has_usable_metadata(rec)}

    print("pass 2/4: review counts per product")
    counts = Counter(rec.get("parent_asin") for rec in stream(RAW_REVIEWS))
    eligible = sorted(p for p, n in counts.items()
                      if n >= c["min_reviews_per_product"] and p in usable)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_candidates = int(c["target_products"] * c["candidate_buffer"])
    candidates = eligible[:n_candidates]
    cand_set = set(candidates)
    print(f"  eligible products: {len(eligible)}; candidate pool drawn: {len(candidates)}")

    print("pass 3/4: collecting reviews of candidate products")
    raw = defaultdict(list)
    for rec in stream(RAW_REVIEWS):
        p = rec.get("parent_asin")
        if p in cand_set:
            raw[p].append(rec)

    stats = Counter()
    chosen, reviews_out, taken_ids = [], [], set()
    for p in candidates:
        recs = sorted(raw[p], key=lambda r: (r.get("timestamp") or 0,
                                             r.get("user_id") or "",
                                             r.get("text") or ""))
        kept, seen = [], set()
        for r in recs:
            text = (r.get("text") or "").strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            key = normalize(text)
            if key in seen:
                stats["dropped_exact_duplicate"] += 1
                continue
            seen.add(key)
            kept.append(r)
        stats["raw_reviews_of_processed_products"] += len(recs)
        if len(kept) < c["min_reviews_per_product"]:
            stats["products_skipped_too_few_after_cleaning"] += 1
            continue
        if len(kept) > c["max_reviews_per_product"]:
            kept = random.Random(f"{seed}-{p}").sample(kept, c["max_reviews_per_product"])
            kept.sort(key=lambda r: (r.get("timestamp") or 0, r.get("user_id") or ""))
        for r in kept:
            text = r["text"].strip()
            reviews_out.append({
                "doc_id": review_doc_id(r, text, taken_ids),
                "parent_asin": p,
                "asin": r.get("asin"),
                "user_id": r.get("user_id"),
                "timestamp": r.get("timestamp"),
                "rating": r.get("rating"),
                "title": (r.get("title") or "").strip(),
                "text": text,
                "helpful_vote": r.get("helpful_vote"),
                "verified_purchase": r.get("verified_purchase"),
            })
        chosen.append(p)
        if len(chosen) >= c["target_products"] and len(reviews_out) >= c["min_total_reviews"]:
            break

    if len(chosen) < c["target_products"] or len(reviews_out) < c["min_total_reviews"]:
        raise SystemExit(
            f"[ERROR] corpus targets not met: {len(chosen)} products, "
            f"{len(reviews_out)} reviews. Increase candidate_buffer or lower thresholds.")

    print("pass 4/4: metadata of chosen products")
    chosen_set = set(chosen)
    meta = {}
    for rec in stream(RAW_META):
        p = rec.get("parent_asin")
        if p in chosen_set:
            meta[p] = rec

    split_rng = random.Random(f"{seed}-split")
    order = list(chosen)
    split_rng.shuffle(order)
    n_dev = round(c["dev_pool_fraction"] * len(order))
    pool = {p: ("dev" if i < n_dev else "test") for i, p in enumerate(order)}
    per_product = Counter(r["parent_asin"] for r in reviews_out)

    os.makedirs(OUT_DIR, exist_ok=True)
    products_path = os.path.join(OUT_DIR, "products.jsonl")
    reviews_path = os.path.join(OUT_DIR, "reviews.jsonl")
    with open(products_path, "w", encoding="utf-8") as fh:
        for p in chosen:
            m = meta[p]
            fh.write(json.dumps({
                "parent_asin": p,
                "title": (m.get("title") or "").strip(),
                "store": m.get("store"),
                "main_category": m.get("main_category"),
                "price": m.get("price"),
                "average_rating": m.get("average_rating"),
                "rating_number": m.get("rating_number"),
                "categories": m.get("categories"),
                "features": str_list(m.get("features")),
                "description": str_list(m.get("description")),
                "details": m.get("details") if isinstance(m.get("details"), dict) else {},
                "split_pool": pool[p],
                "reviews_raw_count": counts[p],
                "reviews_in_corpus": per_product[p],
            }, ensure_ascii=False) + "\n")
    with open(reviews_path, "w", encoding="utf-8") as fh:
        for r in reviews_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "seed": seed,
        "parameters": c,
        "raw_reviews_total": sum(counts.values()),
        "products_with_reviews_raw": len(counts),
        "eligible_products": len(eligible),
        "products_in_corpus": len(chosen),
        "reviews_in_corpus": len(reviews_out),
        "dev_pool_products": n_dev,
        "test_pool_products": len(chosen) - n_dev,
        "cleaning": dict(stats),
        "rating_distribution_in_corpus": {
            str(k): v for k, v in sorted(Counter(r["rating"] for r in reviews_out).items())},
        "reviews_jsonl_md5": md5_file(reviews_path),
        "products_jsonl_md5": md5_file(products_path),
    }
    with open(os.path.join(OUT_DIR, "corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
