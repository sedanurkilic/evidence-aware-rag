#!/usr/bin/env python3
"""LLM access for the RAG experiments: one fixed Gemini model, on-disk cache.

Design rules (from the research plan):
  * One model for every system (no-retrieval, RAG, experiment variants).
  * Every real API call is logged to outputs/llm_calls.jsonl (also failures), so
    the report can state call counts and failed runs.
  * A call that already succeeded is never repeated: responses are cached by a
    hash of (model, temperature, max tokens, system prompt, prompt).
  * Quota exhaustion (HTTP 429) stops the run with a clear message instead of
    silently degrading results.

Usage:
    python src/generation.py --smoke        # one tiny call through the cache
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests
import yaml
from dotenv import load_dotenv

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"   # abstention marker the models must emit

SYSTEM_RAG = (
    "You answer questions about Amazon Health & Personal Care products using ONLY "
    "the numbered sources provided. Rules:\n"
    "1. Cite every claim with the source tag in square brackets, e.g. [S2].\n"
    "2. Do not use outside knowledge.\n"
    f"3. If the sources do not contain enough evidence to answer, start your answer "
    f"with {INSUFFICIENT} and briefly say what is missing.\n"
    "4. Keep the answer short (at most 4 sentences)."
)

SYSTEM_NO_RETRIEVAL = (
    "You answer questions about Amazon Health & Personal Care products using only "
    "your own knowledge. If you cannot answer reliably, start your answer with "
    f"{INSUFFICIENT} and briefly say why. Keep the answer short (at most 4 sentences)."
)


class LLMError(RuntimeError):
    pass


class QuotaExceeded(LLMError):
    pass


def load_config(path="configs/baseline.yaml"):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def format_sources(sources):
    """sources: list of dicts with tag, type ('review'|'metadata'), product_title,
    text and optional rating. Product title is shown on every source because
    review text alone does not name the product."""
    lines = []
    for s in sources:
        kind = "product metadata" if s["type"] == "metadata" else "customer review"
        extra = f", rating {s['rating']}" if s.get("rating") is not None else ""
        lines.append(f"[{s['tag']}] ({kind}{extra}) Product: {s['product_title']}\n{s['text']}")
    return "\n\n".join(lines)


def rag_prompt(question, sources):
    return f"Sources:\n\n{format_sources(sources)}\n\nQuestion: {question}\nAnswer:"


def no_retrieval_prompt(question):
    return f"Question: {question}\nAnswer:"


class LLMClient:
    def __init__(self, cfg):
        load_dotenv()
        self.key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set (put it in .env).")
        g = cfg["generation"]
        self.model = g["model"]
        self.temperature = g["temperature"]
        self.max_tokens = g["max_output_tokens"]
        self.min_interval = g["min_seconds_between_calls"]
        self.cache_dir = g["cache_dir"]
        self.log_path = g["call_log"]
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._last_call = 0.0
        self.calls_made = 0
        self.cache_hits = 0

    # ----------------------------------------------------------------------
    def _cache_key(self, system, prompt):
        payload = json.dumps(
            [self.model, self.temperature, self.max_tokens, system, prompt],
            ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _log(self, entry):
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    # ----------------------------------------------------------------------
    def generate(self, system, prompt, tag=""):
        """Return a record dict. Raises QuotaExceeded / LLMError on failure."""
        key = self._cache_key(system, prompt)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                record = json.load(fh)
            self.cache_hits += 1
            record = dict(record, cache_hit=True)
            return record

        url = f"{API_ROOT}/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        headers = {"x-goog-api-key": self.key, "Content-Type": "application/json"}

        last_error = None
        for attempt in range(1, 4):
            self._throttle()
            started = time.time()
            self._last_call = started
            status, data, err = None, None, None
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=90)
                status = resp.status_code
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                err = f"{exc.__class__.__name__}: {exc}"
            latency = round(time.time() - started, 2)
            self.calls_made += 1

            log = {"ts": datetime.now(timezone.utc).isoformat(), "tag": tag,
                   "model": self.model, "attempt": attempt, "http_status": status,
                   "latency_s": latency, "cache_key": key}

            if status == 200 and isinstance(data, dict):
                cands = data.get("candidates") or []
                parts = (cands[0].get("content", {}).get("parts", []) if cands else [])
                text = "".join(p.get("text", "") for p in parts).strip()
                record = {
                    "cache_key": key, "tag": tag,
                    "model_requested": self.model,
                    "model_version": data.get("modelVersion"),
                    "text": text,
                    "finish_reason": cands[0].get("finishReason") if cands else "NO_CANDIDATES",
                    "block_reason": (data.get("promptFeedback") or {}).get("blockReason"),
                    "usage": data.get("usageMetadata"),
                    "latency_s": latency,
                    "timestamp_utc": log["ts"],
                    "cache_hit": False,
                }
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, indent=2)
                self._log(dict(log, status="ok", finish_reason=record["finish_reason"],
                               model_version=record["model_version"]))
                return record

            message = err or (data or {}).get("error", {}).get("message", "")[:300]
            self._log(dict(log, status="failed", error=message))
            last_error = f"HTTP {status}: {message}" if status else message

            if status == 429:
                if attempt < 3:
                    time.sleep(30)        # per-minute limit may clear; daily will not
                    continue
                raise QuotaExceeded(
                    f"Gemini quota exhausted (HTTP 429): {message}. Stop and decide "
                    "how to continue; do not retry blindly.")
            if status in (400, 401, 403, 404):
                raise LLMError(f"Gemini request rejected ({last_error}).")
            time.sleep(5 * attempt)       # 5xx / network: short back-off

        raise LLMError(f"Gemini call failed after 3 attempts ({last_error}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    client = LLMClient(cfg)
    if args.smoke:
        rec = client.generate(SYSTEM_NO_RETRIEVAL, no_retrieval_prompt(
            "Reply with the single word: OK"), tag="smoke")
        print("model_version:", rec["model_version"])
        print("text:", rec["text"])
        print("usage:", rec["usage"])
        print("cache_hit:", rec["cache_hit"], "| real calls this run:", client.calls_made)


if __name__ == "__main__":
    main()
