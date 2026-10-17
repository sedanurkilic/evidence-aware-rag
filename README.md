# Evidence-aware RAG on Amazon Health & Personal Care reviews

Dense RAG vs a no-retrieval baseline (same LLM), and an investigation of why dense retrieval misses product metadata.
Read `REPORT.md` first (results, uncertainty, limitations), then `RESEARCH_LOG.md` (decisions and predictions, written as the work went on).

## Requirements
Python 3.12, CPU only. `pip install -r requirements.txt`. Free Gemini API key in `.env` (copy `.env.example`, never commit `.env`).

## Quick check (no API key, no download, no index build)
The fixed corpus (`data/processed/`) and every saved run (`outputs/`) are in the repository, so the reported numbers can be recomputed in seconds:
```bash
pip install -r requirements.txt
python src/validate_eval.py evaluation/test.jsonl test
python src/metrics.py evaluation/test.jsonl outputs/test_exp1/dense_rag.jsonl
```
Expected: `OK`, then `answerable: all_facts 15 / 17` (baseline 12 / 17, no_retrieval 5 / 17 by the regex scorer; manual scores are in `outputs/manual_scores_test.csv`).

## Reproduce from scratch
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env            # then put GEMINI_API_KEY=... into .env

# 1. OPTIONAL, only to rebuild the corpus: raw data from Amazon Reviews 2023 (McAuley Lab), category Health_and_Personal_Care, into data/raw/ (compressed files, about 1 GB together):
#      https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Health_and_Personal_Care.jsonl.gz
#      https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Health_and_Personal_Care.jsonl.gz
#    Without it you can start at step `python src/index.py`, because data/processed/ is already in the repository.
mkdir -p data/raw
#   data/raw/Health_and_Personal_Care.jsonl.gz
#   data/raw/meta_Health_and_Personal_Care.jsonl.gz

python src/profile_data.py                    # outputs/data_profile.json
python src/data.py                            # data/processed/{products,reviews}.jsonl, corpus_stats.json (seed 42)
python src/index.py                           # .chroma index, outputs/index_stats.json (about 11 min on an 8 GB laptop CPU)
python src/validate_eval.py evaluation/dev.jsonl dev
python src/validate_eval.py evaluation/test.jsonl test

# development runs
python src/run.py --split dev --system dense_rag
python src/run.py --split dev --system no_retrieval
python src/run.py --split dev --system dense_rag --config configs/experiment_1.yaml --out-dir outputs/dev_exp1
python src/run.py --split dev --system dense_rag --config configs/experiment_2.yaml --out-dir outputs/dev_exp2

# final test (code, prompts, configs frozen at git tag frozen-before-test)
python src/run.py --split test --system dense_rag --confirm-frozen
python src/run.py --split test --system no_retrieval --confirm-frozen
python src/run.py --split test --system dense_rag --config configs/experiment_1.yaml --out-dir outputs/test_exp1 --confirm-frozen
```

## Recompute every metric without any new LLM call
All answers, retrieved chunks (with text), prompts' source tags, model version, usage and configs are stored in `outputs/`.
```bash
python src/metrics.py evaluation/test.jsonl outputs/test/dense_rag.jsonl
python src/metrics.py evaluation/test.jsonl outputs/test_exp1/dense_rag.jsonl
python src/metrics.py evaluation/test.jsonl outputs/test/no_retrieval.jsonl
python src/metrics.py evaluation/test.jsonl outputs/test_exp1/dense_rag.jsonl --show   # also prints the answers
```
Re-running `run.py` also costs no new call for prompts that already have a response in `outputs/llm_cache/` (keyed by model, settings and prompt hash).

## Layout
`configs/` baseline and experiment settings; `src/` code; `evaluation/` dev (10) and test (20) questions with expected facts and evidence ids;
`outputs/` run records, call log (`llm_calls.jsonl`), index and corpus statistics; `data/processed/` the fixed corpus (1,000 products, 27,523 reviews).
The test set is only run with `--confirm-frozen`.

## Disclosure
Developed with an AI assistant (Claude, Anthropic) in a chat session, in dialogue with the author. No retrieval framework (LangChain etc.) is used; libraries: ChromaDB, sentence-transformers, PyYAML, requests, python-dotenv.
