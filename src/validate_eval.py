#!/usr/bin/env python3
"""Check an evaluation file against the corpus (ids exist, split/pool match, schema).

Usage: python src/validate_eval.py evaluation/dev.jsonl dev
"""
import json
import re
import sys

REQUIRED = ["id", "split", "category", "question", "expected_answer", "expected_facts",
            "evidence_doc_ids", "required_source_types", "expected_behavior", "product_asins"]
CATEGORIES = {"metadata", "review", "review_metadata", "comparison", "insufficient"}


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(path, split):
    products = {p["parent_asin"]: p for p in read("data/processed/products.jsonl")}
    doc_ids = {r["doc_id"] for r in read("data/processed/reviews.jsonl")}
    doc_ids |= {f"meta_{a}" for a in products}
    qs, errors = read(path), []
    seen = set()
    for q in qs:
        for k in REQUIRED:
            if k not in q:
                errors.append(f"{q.get('id')}: missing {k}")
        if q["id"] in seen:
            errors.append(f"{q['id']}: duplicate id")
        seen.add(q["id"])
        if q["split"] != split:
            errors.append(f"{q['id']}: split {q['split']} != {split}")
        if q["category"] not in CATEGORIES:
            errors.append(f"{q['id']}: bad category")
        if q["expected_behavior"] not in ("answer", "abstain"):
            errors.append(f"{q['id']}: bad expected_behavior")
        for a in q["product_asins"]:
            if a not in products:
                errors.append(f"{q['id']}: unknown product {a}")
            elif products[a]["split_pool"] != split:
                errors.append(f"{q['id']}: product {a} is in the {products[a]['split_pool']} pool (leakage)")
        for d in q["evidence_doc_ids"]:
            if d not in doc_ids:
                errors.append(f"{q['id']}: unknown doc {d}")
            else:
                asin = d.split("_")[1]
                if asin not in q["product_asins"]:
                    errors.append(f"{q['id']}: doc {d} belongs to a product not in product_asins")
        for fact in q["expected_facts"]:
            for rx in fact:
                re.compile(rx)
        if q["expected_behavior"] == "answer" and not q["evidence_doc_ids"]:
            errors.append(f"{q['id']}: answerable question without evidence")
    cats = {}
    for q in qs:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"{len(qs)} questions in {path}; categories: {cats}")
    for e in errors:
        print("[ERROR]", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
