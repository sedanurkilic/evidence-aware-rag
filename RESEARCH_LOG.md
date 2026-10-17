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

## 2. Sources (read on 2026-10-17; figures were extracted from the arXiv PDFs with an automated reader and should be spot-checked against the papers before being quoted elsewhere)

**S1. Lewis et al., 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — https://arxiv.org/abs/2005.11401
- Finding: a dense retriever (DPR bi-encoder over 21M Wikipedia passages) plus a seq2seq generator (BART-large), trained end to end, beats a closed-book generator on open-domain QA; swapping the index changes what the model knows (about 70% / 68% correct with a matching index, 12% / 4% with a mismatched one); human raters judged RAG output more factual than BART (42.7% vs 7.1%).
- Applicability here: supports the premise of the comparison, since the no-retrieval baseline cannot know product facts that live only in this corpus (seen on dev: 6/9 abstentions and one contradicting hallucination).
- Does not transfer: the paper trains retriever and generator on one homogeneous Wikipedia index; here the retriever is frozen (no training), the LLM is only prompted, and the index mixes two source types (reviews and product metadata) with very different sizes. The paper says nothing about source-type imbalance, which is the problem H1 targets.

**S2. Reimers & Gurevych, 2019, Sentence-BERT** — https://arxiv.org/abs/1908.10084
- Finding: siamese/triplet fine-tuning of BERT with mean pooling gives fixed-size embeddings comparable by cosine similarity, which makes retrieval cheap (finding the most similar pair among 10,000 sentences: about 65 hours with BERT cross-encoding vs about 5 seconds with SBERT embeddings). Cross-encoders are more accurate on supervised STS-B; SBERT was weaker on the cross-topic argument task.
- Applicability here: justifies a bi-encoder with cosine similarity over 29,852 chunks on a CPU. The paper measures semantic similarity, not answer relevance, which matches what we see on dev: chunks that mention the product title score high regardless of whether they contain the asked fact (H2).
- Does not transfer: the paper does not evaluate `all-MiniLM-L6-v2` (a later model of the same family, used here for size); it is trained on NLI/Wikipedia-style data, not on product reviews or "Key: value" metadata, so its behaviour on this domain is untested by the paper (H4, deferred).

**S3. Liu et al., 2024, Lost in the Middle: How Language Models Use Long Contexts** — https://arxiv.org/abs/2307.03172
- Finding: in multi-document QA, accuracy is highest when the relevant document is first or last and drops when it is in the middle (for GPT-3.5-Turbo with 20 documents, 75.8% with the answer first, 53.8% in the worst middle position, 63.2% last); reader accuracy saturates before retriever recall: going from 20 to 50 retrieved documents added only about 1.5 points.
- Applicability here: supports keeping the number of chunks fixed (k=5, the context budget in Experiment 1) rather than raising k (H3 deferred), and warns that where a chunk sits in the prompt can matter. Experiment 1 re-sorts chunks by score, so a metadata chunk can land in the middle; we did not vary position and cannot say whether it matters.
- Does not transfer: the paper uses 10-30 passages with older models; our context has 5 short chunks and a different model, so position effects are likely much smaller here and our n is too small to detect them.

## 3. Development-set baseline results
Dev set: 10 questions (2 metadata, 2 review, 4 review+metadata, 1 comparison, 1 insufficient evidence), built
from dev-pool products only (no product appears in the test set). Evidence ids were checked by the author
against `reviews.jsonl` / `products.jsonl` (`src/validate_eval.py` checks ids and pool membership).
All numbers are counts out of n=9 answerable questions (+1 unanswerable); n is far too small for percentages.

**Retrieval (dense, k=5, top-5 chunks):** gold evidence hit 6/9; joint success (all required source types present) 4/9;
product match 0.98. Metadata-dominant questions (dev_md_01, dev_md_02, dev_cmp_01): metadata document
retrieved 0/3. Metadata chunks appear in the top-5 for only 4/10 questions. The right product is almost always
found; the missing piece is the product's metadata record.

**Answers (regex fact scoring + manual reading):**
| system | all expected facts | false abstentions | correct abstention (1 unanswerable) |
|---|---|---|---|
| dense_rag | 4/9 (regex, before rule fix) | 2 | 1/1 |
| no_retrieval | 1/9 | 6 | 1/1 |

Manual reading of dense_rag answers:
- dev_rev_01 and dev_rm_02 were scored 0 / partial only because my regexes were too narrow ("stops working", "mild orange"
  were not covered). Rules were corrected AFTER seeing these dev answers; the correction is for scoring bugs, applies to dev only,
  and the test rubric is frozen before any test run.
- dev_md_01: answer says "12 ounces per can, pack of 6" but never states the total of 72 oz; the cited sources were reviews, not the metadata record.
  Partially correct, derived from the product title.
- dev_md_02 and dev_cmp_01: the system abstained. Given what was retrieved (no metadata record) this is the intended grounded behaviour;
  the failure is in retrieval, not generation.
- dev_rm_01: answer says "Yes, fragrance-free" (metadata) and that reviewers say it is "free of scents". Checked by hand: retrieved review S3
  (rev_B07XF83W9Y_d9402fc2) does say "free of scents!", so the answer is faithful to what was retrieved. It misses the conflicting reviews
  (scent is nice / smells good / strong odor) because retrieval did not return them. My earlier suspicion of an unfaithful summary was wrong.
- dev_rev_01: answer is "mixed" and includes an asthma complaint [S1] that is not in my gold list. The gold list is therefore not exhaustive;
  evidence recall under-counts valid retrieval for review questions.
- Question-design flaw found after the run: dev_md_01 copies the product title ("12oz Can (Pack of 6)"), so half of the answer is in the question;
  no_retrieval gave the same answer. Kept in the dev set (documented, not silently removed); test questions must not contain their own answer.
- no_retrieval: 6/9 answerable questions abstained ("I do not have access ..."), the rest answered from general knowledge. On dev_rm_01 it asserted that the
  product is "not entirely fragrance-free ... citrus or herbal aroma", which contradicts the product metadata (Scent: Unscented): a hallucination.
  Its dev_rev_02 answer ("adhesive can be weak ... especially on oily skin") matches the regex but is generic, not corpus-based.
- Known metric limits: regex facts are a proxy; no_retrieval's one "success" (dev_rev_02) must be read by hand (a generic answer can
  match the regex without knowing the corpus); citation_match shows relevance of a cited source, not faithfulness.


## 4. Hypotheses (recorded BEFORE any intervention)
Written after the dev baseline and before any change to retrieval or prompts.

| # | Hypothesis | Expected effect if true | How to test | What would challenge it |
|---|-----------|-------------------------|-------------|--------------------------|
| H1 | Source-type imbalance: metadata is 1,480 of 29,852 chunks (5%) and its chunks score lower than reviews for questions that name the product, so metadata rarely reaches the top-5 | Reserving part of the fixed top-5 for metadata raises metadata hit and joint success, and fixes metadata-dependent answers | Experiment 1: source-aware retrieval at the same k=5 and same context budget (e.g. top-3 reviews + top-2 metadata) vs unchanged baseline | Metadata is retrieved but answers still fail, or review-only questions regress strongly |
| H2 | Title dominance: the product title in every review chunk makes any review of the named product match the query regardless of content | Queries naming a product return content-free reviews; removing the title from review chunks changes ranking and lowers product match | Re-index reviews without the title header, compare retrieved chunks | Ranking and scores barely change |
| H3 | Top-k is too small | k=10 brings metadata into the context | Run k=10 | Metadata is still absent at k=10 (then H1 is stronger), or answers get worse from a longer context (lost-in-the-middle) |
| H4 | MiniLM represents "Key: value" metadata text poorly for factual questions | A different embedding model ranks metadata higher | Swap the embedding model, re-index | Same failures with another model |
| H5 | The LLM fails to use or over-abstains on available evidence | Even with perfect (oracle) evidence in the prompt answers stay wrong | Oracle-context run on dev | Oracle context gives high accuracy (generation is not the bottleneck) |

**Selected: H1.** Reasons: it explains the clearest dev pattern (metadata record missing in 3/3 metadata-dominant questions and in the joint failures of
dev_rm_01 and dev_rm_02), it can be tested without re-indexing, and with a fixed k and context budget it changes one thing.
**Deferred:** H2 and H4 need a re-index (about 11 minutes each) and would change the embedding space, so they would not be a single targeted change;
H3 is confounded with context length and cost; H5 is used as the diagnostic in Experiment 2 (below).
Caveat: n=10 dev questions; the pattern may be partly the result of how I wrote them (many ask for structured facts).

## 5. Experiment 1 — selection and prediction
**Change (H1, one thing):** source-aware retrieval. Same index, same embedding, same k=5, same prompts, same LLM. Baseline takes the 5 best chunks
overall; Experiment 1 takes the 3 best review chunks + the 2 best metadata chunks (`configs/experiment_1.yaml`, `retrieval.mode: source_aware`),
merged by score. Output: `outputs/dev_exp1/dense_rag.jsonl`. The baseline is the unchanged run in `outputs/dev/dense_rag.jsonl`,
rescored with the same (corrected) rubric.

**Prediction (written before running):**
1. Metadata record for the right product in the top-5: baseline 4/10 questions -> at least 8/10.
2. Joint success (answerable questions): baseline 4/9 -> at least 7/9.
3. Metadata-dominant questions (md_01, md_02, cmp_01): metadata hit 0/3 -> at least 2/3; false abstentions on them fall.
4. Review-only questions (rev_01, rev_02): evidence recall per question may FALL (3 review slots instead of 5); I expect no change in hit (2/2) but possibly fewer gold reviews found.
5. dev_ins_01 stays a correct abstention.
6. Risk: dev_cmp_01 needs both products' metadata with only 2 slots; two chunks of one product could use both slots. Expected to succeed in at most 1 of the 2 products if that happens.
**What would count against H1:** metadata is retrieved in >=8/10 questions but all-facts accuracy does not improve, or review-only questions regress.

### Experiment 1 — result (dev, n=9 answerable + 1 unanswerable; baseline rescored with the same corrected rubric)
| | baseline (dense) | Exp 1 (source-aware) |
|---|---|---|
| gold evidence hit@5 | 6/9 | 9/9 |
| joint success (needed source types) | 4/9 | 8/9 |
| metadata-dominant questions with metadata retrieved | 0/3 | 3/3 |
| all expected facts in answer (regex) | 6/9 | 9/9 |
| false abstentions | 2 | 0 |
| correct abstention on the unanswerable question | 1/1 | 1/1 |
| mean product match of the 5 chunks | 0.98 | 0.84 |
| gold reviews found (dev_rev_02) | 2/6 | 1/6 |
Prediction check: (1) metadata for the right product: measured via hit/joint, consistent; (2) joint >=7/9: 8/9, met; (3) >=2/3 metadata-dominant: 3/3, met;
(4) review-only recall may fall: dev_rev_02 fell 2/6 -> 1/6, hit stayed 2/2, met; (5) abstention kept, met; (6) risk for dev_cmp_01 did not materialise (both products' metadata retrieved).
Costs / caveats: the forced metadata slots bring in other products' metadata (product match 0.98 -> 0.84); dev_rm_01 still has joint=0 (the gold reviews were not retrieved
although the metadata was). n is tiny, H1 was chosen and evaluated on the same 10 dev questions, dev_md_01 is a flawed question; the test set is the real check.

## 6. Experiment 2 — prediction recorded before running
**Question:** is the Exp 1 gain due to the RIGHT PRODUCT's metadata reaching the context (explanation E), or just to a changed context composition / extra
metadata-type chunks whatever product they describe (alternative A)?
**Diagnostic (dev only, uses gold product ids, not a deployable system, never run on test):** same as Exp 1, but the 2 metadata slots are filled with the best
metadata chunks of OTHER products (`diagnostic_exclude_gold_metadata`, `configs/experiment_2.yaml`). Review slots unchanged. Output `outputs/dev_exp2/`.
**Prediction (before running):**
- Metadata-dependent answerable questions: 7 (md_01, md_02, cmp_01, rm_01..rm_04). Baseline all-facts on them: 4/7; Exp 1: 7/7.
- If E is right: with wrong-product metadata, all-facts on these 7 falls to **<=4/7** (about baseline level or below), with abstentions or review-only answers on md_01, md_02, cmp_01.
  Review-only questions (rev_01, rev_02) stay correct.
- If A is right: all-facts stays **>=6/7** even with wrong-product metadata.
- 5/7 is an inconclusive result and will be reported as such.
- My expectation: E. Abstention on dev_ins_01 remains correct under both.
**What would challenge E:** all-facts >=6/7 with wrong-product metadata.
(Correction note: the first draft of this prediction used "6 questions / baseline 3/6" by a counting mistake; it was fixed before Experiment 2 was run.)

### Experiment 2 — result (dev; diagnostic, uses gold product ids)
Metadata slots filled with the best metadata chunks of OTHER products; review slots unchanged.
| question | baseline | Exp 1 | Exp 2 (wrong-product metadata) |
|---|---|---|---|
| dev_md_01 | wrong (abstain-free, partial) | ok | abstain (no "unit count" found) |
| dev_md_02 | abstain | ok | abstain (material only, no hose length) |
| dev_cmp_01 | abstain | ok | abstain (no Maybelline rating) |
| dev_rm_01 | ok (regex) | ok | missed "fragrance free" claim; said manufacturer claim is not stated |
| dev_rm_02 | ok | ok | ok |
| dev_rm_03 | ok | ok | abstain (no capsule count) |
| dev_rm_04 | ok | ok | ok |
| **all facts on the 7 metadata-dependent questions** | **4/7** | **7/7** | **2/7** |
Review-only questions (rev_01, rev_02) stayed correct (2/2); the unanswerable question stayed a correct abstention.
Prediction check: E predicted <=4/7, A predicted >=6/7; observed 2/7 -> consistent with E, against A.
Reading: the right product's metadata is needed for the metadata facts. The behaviour with missing metadata is the intended one (the answers say what the sources do not contain).
Limits of this diagnostic: (a) it shows the right metadata is NECESSARY; it cannot separate "missing right metadata" from "distraction by wrong metadata" because both change together;
(b) dev_rm_02 and dev_rm_04 still passed because the product title that every review chunk carries already contains "Rechargeable" and "8 oz (234g)" — the title header (H2) leaks metadata
into review chunks, so some facts are answerable from titles; (c) n=10, dev only, one run; (d) the diagnostic uses gold labels and is not a usable system and is not run on test.

## 7. Final test run
_(to be filled; test set opened only after configurations are frozen)_
