#!/usr/bin/env python3
"""Profile the raw Amazon Reviews 2023 (Health & Personal Care) files.

Purpose: decide the sampling rule from measured numbers, not guesses.
Streams both gzip files (low memory), prints a summary and saves it to
outputs/data_profile.json. Uses only the standard library.

Usage (from the repository root):
    python src/profile_data.py
"""

import gzip
import hashlib
import json
import os
import statistics
from collections import Counter

REVIEWS = os.path.join("data", "raw", "Health_and_Personal_Care.jsonl.gz")
META = os.path.join("data", "raw", "meta_Health_and_Personal_Care.jsonl.gz")
OUT = os.path.join("outputs", "data_profile.json")
THRESHOLDS = [5, 10, 15, 20, 25, 30, 50]


def stream(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(values, q):
    """q-th percentile (0-100) of a non-empty list, nearest-rank."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]


def words(text):
    return len(text.split()) if isinstance(text, str) else 0


def meta_text_len(rec):
    parts = []
    for key in ("features", "description"):
        val = rec.get(key)
        if isinstance(val, list):
            parts.extend(str(x) for x in val)
    details = rec.get("details")
    if isinstance(details, dict):
        parts.extend(f"{k}: {v}" for k, v in details.items())
    return sum(words(p) for p in parts), bool(parts)


def main():
    per_product = Counter()
    ratings = Counter()
    review_words = []
    empty_text = 0
    seen = set()
    duplicates = 0
    total = 0
    keys_seen = None

    for rec in stream(REVIEWS):
        total += 1
        if keys_seen is None:
            keys_seen = sorted(rec.keys())
        per_product[rec.get("parent_asin")] += 1
        ratings[rec.get("rating")] += 1
        text = rec.get("text") or ""
        if not text.strip():
            empty_text += 1
        review_words.append(words(text))
        digest = hashlib.md5(
            f"{rec.get('parent_asin')}|{text.strip()}".encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
        else:
            seen.add(digest)

    meta_ok = {}          # parent_asin -> has usable metadata (title + some text)
    meta_words = []
    meta_total = 0
    meta_keys = None
    for rec in stream(META):
        meta_total += 1
        if meta_keys is None:
            meta_keys = sorted(rec.keys())
        n_words, has_text = meta_text_len(rec)
        usable = bool((rec.get("title") or "").strip()) and has_text
        meta_ok[rec.get("parent_asin")] = usable
        if has_text:
            meta_words.append(n_words)

    products_with_reviews = len(per_product)
    overlap = sum(1 for p in per_product if p in meta_ok)
    usable_overlap = sum(1 for p in per_product if meta_ok.get(p))

    counts = list(per_product.values())
    by_threshold = {}
    for n in THRESHOLDS:
        prods = [p for p, c in per_product.items() if c >= n and meta_ok.get(p)]
        by_threshold[str(n)] = {
            "products_with_usable_metadata": len(prods),
            "reviews_in_those_products": sum(per_product[p] for p in prods),
        }

    summary = {
        "review_fields": keys_seen,
        "meta_fields": meta_keys,
        "reviews_total": total,
        "meta_records_total": meta_total,
        "products_with_reviews": products_with_reviews,
        "products_with_reviews_and_meta": overlap,
        "products_with_reviews_and_usable_meta": usable_overlap,
        "reviews_per_product": {
            "mean": round(statistics.mean(counts), 2),
            "median": statistics.median(counts),
            "p90": pct(counts, 90),
            "p99": pct(counts, 99),
            "max": max(counts),
        },
        "thresholds": by_threshold,
        "review_words": {
            "median": statistics.median(review_words),
            "p90": pct(review_words, 90),
            "p99": pct(review_words, 99),
            "share_over_200_words": round(
                sum(1 for w in review_words if w > 200) / total, 4),
        },
        "meta_text_words": {
            "median": statistics.median(meta_words),
            "p90": pct(meta_words, 90),
            "p99": pct(meta_words, 99),
        } if meta_words else None,
        "empty_review_text": empty_text,
        "exact_duplicate_reviews_within_product": duplicates,
        "rating_distribution": {str(k): v for k, v in sorted(
            ratings.items(), key=lambda kv: (kv[0] is None, kv[0]))},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()