#!/usr/bin/env python3
"""Recompute metrics from saved run outputs (no LLM calls, no retrieval calls).

Retrieval metrics (answerable questions only; evidence compared at source-document level):
  evidence_hit      : at least one gold evidence document is in the top-k
  evidence_recall   : share of gold evidence documents found in the top-k
  joint_success     : for EVERY required source type, at least one gold evidence
                      document of that type is in the top-k (the answer needs both)
  product_match     : share of top-k chunks that belong to the question's product(s)

Answer metrics (need a run with generated answers):
  fact_acc          : share of expected facts found in the answer (regex alternatives,
                      case-insensitive; a deterministic proxy, spot-check by hand with --show)
  all_facts         : every expected fact found
  abstain_ok        : expected abstain -> answer starts with INSUFFICIENT_EVIDENCE
  false_abstain     : answerable question but the system abstained
  citation_match    : at least one cited [S#] chunk is a gold evidence document.
                      This is NOT faithfulness: it only shows a cited source is relevant.

Usage:
    python src/metrics.py evaluation/dev.jsonl outputs/dev/dense_rag_retrieval_only.jsonl
    python src/metrics.py evaluation/dev.jsonl outputs/dev/dense_rag.jsonl [--show]
"""
import json
import re
import sys
from collections import defaultdict


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def retrieval_scores(q, rec):
    docs = [h["doc_id"] for h in rec["retrieved"]]
    gold = set(q["evidence_doc_ids"])
    found = gold & set(docs)
    types_ok = all(
        any(d in gold and (d.startswith("meta_") == (t == "metadata")) for d in docs)
        for t in q["required_source_types"])
    prod = [h["parent_asin"] in q["product_asins"] for h in rec["retrieved"]]
    return {"evidence_hit": bool(found), "evidence_recall": len(found) / len(gold),
            "joint_success": types_ok, "product_match": sum(prod) / max(len(prod), 1),
            "n_found": len(found), "n_gold": len(gold)}


def answer_scores(q, rec):
    ans = rec["answer"] or ""
    hits = [any(re.search(rx, ans, re.I) for rx in fact) for fact in q["expected_facts"]]
    gold = set(q["evidence_doc_ids"])
    by_tag = {f"S{h['rank']}": h["doc_id"] for h in rec["retrieved"]}
    cited_gold = any(by_tag.get(t) in gold for t in rec["cited_tags"])
    out = {"abstained": bool(rec["abstained"]), "citation_match": cited_gold,
           "n_cited": len(rec["cited_tags"])}
    if q["expected_behavior"] == "abstain":
        out["abstain_ok"] = bool(rec["abstained"])
    else:
        out["false_abstain"] = bool(rec["abstained"])
        out["fact_acc"] = sum(hits) / len(hits) if hits else 0.0
        out["all_facts"] = all(hits) and not rec["abstained"]
    return out


def main_answers(qs, recs, show):
    rows = []
    for qid, q in qs.items():
        if qid in recs and recs[qid].get("answer") is not None:
            rows.append((q, recs[qid], answer_scores(q, recs[qid])))
    print(f"{'id':<12}{'category':<17}{'facts':<8}{'all':<5}{'abst':<6}{'cite_match'}")
    for q, rec, a in rows:
        facts = f"{a['fact_acc']:.2f}" if "fact_acc" in a else "-"
        allf = str(int(a["all_facts"])) if "all_facts" in a else "-"
        cm = int(a["citation_match"]) if rec["retrieved"] else "n/a"
        print(f"{q['id']:<12}{q['category']:<17}{facts:<8}{allf:<5}{int(a['abstained']):<6}{cm}")
        if show:
            print("   Q:", q["question"]); print("   A:", (rec["answer"] or "").replace("\n", " ")); print()
    ans = [(q, a) for q, _, a in rows if q["expected_behavior"] == "answer"]
    print("\nanswerable: all_facts", sum(a["all_facts"] for _, a in ans), "/", len(ans),
          "| false abstentions", sum(a["false_abstain"] for _, a in ans))
    ins = [a for q, _, a in rows if q["expected_behavior"] == "abstain"]
    print("unanswerable: correct abstentions", sum(a["abstain_ok"] for a in ins), "/", len(ins))
    cats = defaultdict(list)
    for q, a in ans:
        cats[q["category"]].append(a)
    for c, ss in cats.items():
        print(f"  {c:<17}all_facts {sum(x['all_facts'] for x in ss)}/{len(ss)}")


def main(eval_path, run_path, show=False):
    qs = {q["id"]: q for q in read(eval_path)}
    recs = {r["id"]: r for r in read(run_path)}
    if any(r.get("answer") is not None for r in recs.values()):
        return main_answers(qs, recs, show)
    rows = []
    for qid, q in qs.items():
        if q["expected_behavior"] != "answer" or qid not in recs:
            continue
        rows.append((q["category"], qid, retrieval_scores(q, recs[qid])))
    print(f"{'id':<12}{'category':<17}{'found/gold':<11}{'hit':<5}{'joint':<7}{'prod_match@k'}")
    for cat, qid, s in rows:
        print(f"{qid:<12}{cat:<17}{s['n_found']}/{s['n_gold']:<9}{int(s['evidence_hit']):<5}"
              f"{int(s['joint_success']):<7}{s['product_match']:.2f}")
    groups = defaultdict(list)
    for cat, _, s in rows:
        groups[cat].append(s)
        groups["ALL"].append(s)
    print("\ncounts (hit / joint / n):")
    for cat, ss in groups.items():
        print(f"  {cat:<17}{sum(s['evidence_hit'] for s in ss)} / "
              f"{sum(s['joint_success'] for s in ss)} / {len(ss)}   "
              f"mean product_match {sum(s['product_match'] for s in ss)/len(ss):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--show" in sys.argv)