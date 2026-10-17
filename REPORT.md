# REPORT: When retrieval misses the connection — source-type imbalance in product-review RAG

Author: Seda Nur Kılıç. Date: 2026-10-17. Developed with an AI assistant (Claude, Anthropic), see Disclosure. Decisions, predictions and diagnostics are in `RESEARCH_LOG.md`.

## 1. Summary
I built a dense RAG system over Amazon Reviews 2023 (Health & Personal Care) with two source types in one index (reviews and product metadata) and compared it with a no-retrieval baseline using the same LLM. On a 10-question dev set the baseline retrieved the right product but rarely its metadata record, so metadata questions failed. I selected the hypothesis "source-type imbalance" (metadata is 5% of chunks and scores lower than reviews) and tested one change at a fixed context size: take 3 review chunks and 2 metadata chunks instead of the 5 best overall (Experiment 1). A diagnostic follow-up (Experiment 2, dev only) replaced the metadata with other products' metadata to check that the right product's metadata is what matters. On the frozen test set (20 questions, 17 answerable) Experiment 1 answered all expected facts for 15/17 questions against 12/17 for the baseline (11/17 after a manual correction, see 6) and 5/17 without retrieval. The difference between baseline and Experiment 1 is 3 questions fixed, 0 regressed (exact sign test p = 0.25, so this test set alone does not rule out chance). Two comparison questions stay unsolved, and the intervention lowers product match from 0.89 to 0.78.

## 2. System
- Corpus (`src/data.py`, seed 42): products with at least 20 reviews and usable metadata (4,171 eligible); 1,000 drawn by seed; at most 30 reviews per product; empty and exact-duplicate reviews removed. Result: 1,000 products, 27,523 reviews (raw: 494,121 reviews, 60,293 metadata records). Only products with many reviews qualify, so the corpus is biased toward popular products. Products are assigned to a dev pool (330) or test pool (670); questions for each split use only that pool.
- Index (`src/index.py`): ChromaDB, cosine, `sentence-transformers/all-MiniLM-L6-v2` on CPU, chunks of at most 256 tokens, 29,852 chunks (1,480 of them product metadata). Every chunk starts with the product title (a deliberately fair baseline, since reviews rarely name the product).
- Retrieval: top-5 chunks by cosine similarity.
- Generation: `gemini-3.5-flash-lite` (Gemini API free tier), temperature 0, max 600 output tokens, same model for all systems. The RAG prompt requires `[S#]` citations, forbids outside knowledge and requires the answer to start with `INSUFFICIENT_EVIDENCE` when the sources do not contain the answer; the no-retrieval prompt uses the same marker.

## 3. Evaluation data
30 questions (`evaluation/`): dev 10 (2 metadata, 2 review, 4 review+metadata, 1 comparison, 1 insufficient evidence), test 20 (4, 4, 6, 3, 3). Each has expected facts (regex alternatives), gold evidence document ids, required source types and expected behaviour (answer or abstain). Dev questions use dev-pool products only, test questions test-pool products only (`src/validate_eval.py` checks this: 0 shared products). The test set was written, checked and frozen before any test run; it was not used for tuning.
How they were made: candidates were found by scripts over the processed data (metadata fields, review keyword matches, absence checks) with AI assistance; the author checked the evidence of the dev questions and of the three insufficient-evidence test questions against the data files (absence confirmed in both metadata and all reviews). Gold review lists are capped at six per question and are not exhaustive. Known flaw: dev_md_01 repeats the product title that already contains half of its answer.

## 4. Metrics (all recomputable from saved outputs with `python src/metrics.py`)
- Retrieval, at source-document level: evidence hit (at least one gold document in the top 5), joint success (every required source type present among gold documents), product match (share of the 5 chunks from the question's product).
- Answer quality: all expected facts present (regex proxy), false abstention.
- Grounding: correct abstention on unanswerable questions; citation-evidence match (a cited chunk is a gold document). This is not faithfulness; a correct citation only shows the cited source is relevant.
Because n is small, results are counts, not percentages.

## 5. Sources and hypotheses
Sources read (findings and applicability in `RESEARCH_LOG.md` section 2): Lewis et al. 2020 (RAG), Reimers & Gurevych 2019 (Sentence-BERT), Liu et al. 2024 (Lost in the Middle).
Hypotheses recorded before any intervention (section 4 of the log): H1 source-type imbalance (selected), H2 product-title dominance in chunks, H3 top-k too small, H4 the embedding model represents "Key: value" metadata poorly, H5 the LLM fails to use available evidence. Selected H1 because it explained the clearest dev pattern (metadata record missing in 3/3 metadata-dominant questions) and could be tested at fixed k without re-indexing. H2 and H4 need a re-index and would change the embedding space; H3 is confounded with context length; H5 became the idea behind the Experiment 2 diagnostic (partly: it tests the retriever side, not oracle generation). H2 to H5 were not run.

## 6. Results
**Dev baseline (n=9 answerable):** gold evidence hit 6/9, joint success 4/9, metadata record retrieved for 0/3 metadata-dominant questions. Answers: all facts 6/9 after fixing two regexes that were too narrow (corrected after seeing the dev answers; dev only), no-retrieval 1/9.
**Experiment 1 (dev, prediction in the log written before running):** joint 4/9 -> 8/9; all facts 6/9 -> 9/9; false abstentions 2 -> 0; product match 0.98 -> 0.84; gold reviews found for dev_rev_02 2/6 -> 1/6. Predictions 1 to 5 held; the predicted risk for the comparison question did not appear.
**Experiment 2 (dev, diagnostic using gold labels, not run on test):** with metadata slots filled by other products' metadata, all facts on the 7 metadata-dependent questions fell to 2/7 (baseline 4/7, Experiment 1 7/7); prediction was at most 4/7 if the right product's metadata matters, at least 6/7 if only context composition matters. Result: consistent with "right product's metadata matters". It cannot separate missing right metadata from distraction by wrong metadata.
**Final test (frozen at tag `frozen-before-test`; each system run once; 0 failed questions):**

| system | all expected facts (n=17) | false abstentions | correct abstentions (n=3) |
|---|---|---|---|
| no_retrieval | 5/17 | 12 | 2/3 |
| dense_rag (baseline) | 12/17 (11/17 manually adjusted) | 5 | 3/3 |
| Exp 1 source-aware | 15/17 | 2 | 3/3 |

| category (n) | no_retrieval | baseline | Exp 1 |
|---|---|---|---|
| metadata (4) | 0 | 2 | 4 |
| review (4) | 2 | 4 | 4 |
| review+metadata (6) | 3 | 6 | 6 |
| comparison (3) | 0 | 0 | 1 |

| retrieval (answerable) | evidence hit | joint success | product match |
|---|---|---|---|
| baseline | 12/17 | 7/17 | 0.89 |
| Exp 1 | 17/17 | 15/17 | 0.78 |

Joint success by category, baseline -> Exp 1: metadata 2/4 -> 4/4, review 4/4 -> 4/4, review+metadata 1/6 -> 4/6, comparison 0/3 -> 3/3.
Regressions: none in answer facts on test (3 fixed, 14 unchanged); the costs are lower product match (0.89 -> 0.78) and, on dev, fewer gold reviews found for a review question. Failed runs: none on dev or test; setup attempts that failed before the evaluation (an OpenRouter free quota of 50 calls/day, an unavailable Gemini model id) are in the log.
Manual adjustment: the baseline answer to test_rm_06 says the cream "is not unscented" and was counted correct by the frozen regex (it matches "unscented"), but the product listing says "Unscented"; the baseline had no metadata in its context. I did not change the rubric and report 12/17 (regex) and 11/17 (adjusted). The no-retrieval score cannot tell corpus knowledge from generic plausible text and I did not read all its answers.

## 7. Uncertainty
- n: 17 answerable test questions, 3 unanswerable; one run per system; 3 questions explain the baseline-versus-Experiment 1 difference (sign test p = 0.25). The dev result went the same way (3 questions) but H1 was chosen and evaluated on those 10 questions, so dev is not independent confirmation.
- The questions were written by the author and the assistant in the style of structured product facts; a more review-heavy question mix would probably show a smaller gain.
- Scoring is a regex proxy (one false positive found, see above), gold evidence is capped and not exhaustive, temperature 0 does not guarantee identical LLM outputs, the free-tier model may change.
- The corpus is biased to popular products, and similar products appear in both pools (a dev-pool product of the same brand was retrieved for a test question), so pool separation is by product, not by product family.

## 8. Strongest alternative explanation
The gain may come less from "balancing sources" than from simply adding any product-level record to a context made of near-duplicate review chunks, on a question set that is metadata-heavy. Evidence for this reading: in the baseline, review+metadata questions were answered correctly even with joint success 1/6, because the title in every chunk and the review text already carried facts such as "vegan" and "rechargeable". Experiment 2 supports that the right product's metadata is necessary for metadata facts, but it does not show that a fixed 3+2 split is better than other ways of adding metadata (for example a larger k, H3, which I did not run).

## 9. Three concrete cases
1. Success, test_md_03 ("What flavor and manufacturer are listed for the Grenade Carb Killa ... 12 Count?"). Baseline retrieved only reviews and abstained: "...do not list the specific individual flavors". Experiment 1 retrieved the product's metadata record as S4 and answered "Caramel Chaos ... Grenade USA, LLC [S4]". Retrieval, not generation, was the bottleneck here.
2. Failure not fixed, test_cmp_02 (which of Grenade Carb Killa and Clif Luna Bar has more ratings, and the higher rating). Experiment 1 retrieved three Luna reviews, the Luna metadata record and the metadata of an unrelated protein bar; the Grenade product never appeared and the system abstained. test_cmp_03 failed similarly (Zim's retrieved, Art of Sport not; the retrieved Zim's record was a sibling product from the dev pool). Question embedding favours one named product, so reserving metadata slots does not guarantee both products appear. This is evidence that the selected explanation covers only part of the comparison failures.
3. Evidence against the explanation, test_rm_01/test_rm_06 in the baseline. For test_rm_01 (JYM Plant Jym "vegan?") the baseline answered correctly without the metadata record in context, because the title and reviews carried the fact. For test_rm_06 the baseline, again without the metadata record, said "not unscented" and described reviewer comments about a menthol smell, contradicting the listing ("Unscented"); Experiment 1 reported both: listed as unscented, but reviewers note a menthol or wintergreen smell. So missing metadata can produce a confident wrong answer, but it does not always cause failure, and my joint-success metric overstates how often metadata is needed.

## 10. Next experiment
Per-entity retrieval for multi-product questions (retrieve separately for each named product, reserve metadata per product), prediction written before running: test_cmp_02 and test_cmp_03 are fixed without regressions elsewhere. A second step to separate H1 from H2: rebuild the review chunks without the product title and check whether metadata still fails to reach the top 5.

## 11. Reproducibility
Models: embedding `sentence-transformers/all-MiniLM-L6-v2`; LLM `gemini-3.5-flash-lite` (the served version string is stored per answer in `model_version`). Hardware: MacBook with Apple M2 and 8 GB RAM, CPU only (embedding), Python 3.12.2; versions in `requirements.txt`. Runtime: index build about 651 s (`outputs/index_stats.json`); each 20-question LLM run took about 1.5 to 3 minutes at 4.5 s between calls; all three test runs about 6 minutes. API calls: 99 real Gemini calls logged in `outputs/llm_calls.jsonl` (1 smoke test, 40 dev, 58 test), 2 cache hits, 0 failed. Commands are in `README.md`. Seeds: 42.

## 12. Incomplete work and limitations
- H2, H3, H4 and H5 were not tested; no oracle-generation run; Experiment 2 was dev-only and used gold labels.
- Not every test answer was read by hand; the regex scorer was not validated beyond the cases above; faithfulness was not measured (citation match only).
- No confidence intervals beyond one exact sign test; one run per system.
- Corpus limited to popular products; near-duplicate products across pools were not removed; the free-tier LLM and a small embedding model were used.
- dev_md_01 is a flawed question and stayed in the dev set.

## 13. Disclosure
AI assistance: the code, the question candidates and the experimental design were developed with an AI assistant (Claude, Anthropic) in a chat session, in dialogue with the author, who ran all code and made the recorded decisions. Source-paper figures were extracted with an automated reader and should be spot-checked. No code was copied from other projects; libraries used: ChromaDB, sentence-transformers (with PyTorch and Transformers), PyYAML, requests, python-dotenv.
