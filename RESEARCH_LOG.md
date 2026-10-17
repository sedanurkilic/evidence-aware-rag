# RESEARCH_LOG

Working log, written as decisions are made (not afterwards). Dates: 2026-10-17.
Entries are append-only; a changed decision gets a new entry that references the old one.

## 0. Disclosures
- AI assistance: the code and the experimental design were developed with an AI assistant
  (Claude, Anthropic) in a chat session, in dialogue with the author. The author ran all
  code, made the decisions recorded here, and is responsible for the results.
- Reused libraries: ChromaDB, sentence-transformers, PyYAML, requests, python-dotenv.
  No retrieval framework (LangChain etc.) is used.

## 1. Setup decisions (before any evaluation)

| # | Decision | Reason |
|---|----------|--------|
| D1 | Dataset: Amazon Reviews 2023, Health & Personal Care (reviews + `meta_` file), joined on `parent_asin` | Task requirement; the dataset documentation names `parent_asin` as the review-metadata key |
| D2 | Measured raw size: 494,121 reviews, 60,293 metadata records, 60,274 products with reviews (`outputs/data_profile.json`) | Planning figures from memory were wrong (461.7K); counts were measured, not assumed |
| D3 | Sampling: products with >=20 reviews and usable metadata (4,171 eligible); seeded draw (seed 42) of 1,000 products; max 30 reviews per product (seeded); drop empty and exact-duplicate reviews | The median product has only 2 reviews, so a fixed per-product quota is only possible for popular products. Known limitation: corpus is biased to popular products |
| D4 | Resulting corpus: 1,000 products, 27,523 reviews, 330 dev-pool / 670 test-pool products; dropped 32 empty and 1,781 duplicate reviews; 15 products skipped after cleaning (`data/processed/corpus_stats.json`) | Meets >=25,000 reviews; product-level dev/test pools so questions never share products across splits |
| D5 | Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (256-token limit), CPU | Free, open source, runs on an 8 GB laptop; reviews are short (median 20 words) |
| D6 | Chunks are packed to <=256 tokens including header; every chunk keeps its source `doc_id` | Avoids silent truncation (about 1.5% of reviews exceed 200 words; metadata is longer) |
| D7 | Fair baseline: every chunk starts with the product title | Review text rarely names the product; without the title, product-named questions could never match reviews and the baseline would be a straw man. The same title is shown to the LLM next to each source |
| D8 | Vector DB: ChromaDB (local, cosine) | Free, open source, local |
| D9 | LLM: `gemini-3.5-flash-lite` via the Gemini API free tier, temperature 0, one model for every system | OpenRouter free models were limited to 50 calls/day (measured with `/api/v1/key`), too few for the design. Google AI Studio showed 15 requests/min and 500/day for this model. A fixed-version model id is used (no `-latest` alias) |
| D10 | Abstention is measured through a fixed marker `INSUFFICIENT_EVIDENCE` that both baselines are told to emit | Allows deterministic scoring of "acknowledge insufficient evidence" |
| D11 | Every real LLM call is logged (`outputs/llm_calls.jsonl`) and cached by prompt hash; a successful call is never repeated | Call counts and failed runs must be reportable; quota is limited |

## 2. Sources (to be completed after reading)
Planned primary sources (links verified to exist; findings and applicability to be written after reading):
- Lewis et al., 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — https://arxiv.org/abs/2005.11401
- Reimers & Gurevych, 2019, Sentence-BERT — https://arxiv.org/abs/1908.10084
- Liu et al., 2024, Lost in the Middle: How Language Models Use Long Contexts — https://arxiv.org/abs/2307.03172

## 3. Development-set baseline results
_(to be filled after the dev run)_

## 4. Hypotheses (recorded BEFORE any intervention)
_(to be filled after the dev baseline; at least four, each with expected effect, test and refuting result)_

## 5. Experiment 1 — selection and prediction
_(to be filled)_

## 6. Experiment 2 — prediction recorded before running
_(to be filled)_

## 7. Final test run
_(to be filled; test set opened only after configurations are frozen)_
