# AutoResearch v0 — BM25 keyword search

> **Update: 40-question eval (2026-11-23).** Re-evaluated with all versions on the expanded eval set (q1–q40, 29 new implicit multi-aspect questions): P@10 0.645 · P@50 0.535 · R@50 0.337 · nDCG@50 0.652. v0 is the weakest system on implicit questions (R@50 0.320), where the question's own words rarely match the papers' terms. Full comparison: [`eval_40_results.md`](eval_40_results.md).
<!-- ## Update: 40-question eval -->

v0 is the baseline: plain keyword search over paper titles and abstracts. **There is no LLM
and no neural model.** Every later version is measured against it, and in v2 it becomes a
tool the planner agent calls.

## Quick start

```bash
source .venv/bin/activate
python -m autoresearch build-index                       # once, and after new data is fetched
python -m autoresearch search "structural similarity between stories" -k 20
python -m autoresearch search "..." --abstracts          # also print abstracts
python -m autoresearch search "..." --explain            # show the tokens BM25 matches on
```

## How it works

```
data/raw/*.jsonl ──► Corpus ──► tokenize ──► BM25 index (data/index/bm25/)
                                                   │
query ──► tokenize ──► score every paper ──► top-k ──► print + trace (runs/*.jsonl)
```

### 1. Corpus (`autoresearch/corpus.py`)
- Loads every `data/raw/*.jsonl` file (one paper per line, unified schema produced by
  `scripts/fetch_proceedings.py`).
- Drops duplicate ids and sorts papers by id, so a paper's position (its *doc index*) is the
  same on every build.
- Text that BM25 indexes for each paper: `title + title + abstract`. The title is repeated so
  its words count twice; titles are short and keyword-dense, abstracts are long and diluted.
- Authors, venue, track and links are **not** searched; they are only used for display.

### 2. Tokenization (`autoresearch/bm25.py`)
Applied identically to documents and queries (BM25 only; dense search in v1 will see raw text):

1. plural acronyms → singular, before lowercasing: `LLMs → LLM`, `VLMs → VLM`, `GNNs → GNN`
   (the stemmer leaves `llms` alone, so otherwise "LLMs" and "LLM" never match)
2. lowercase
3. remove stopwords: bm25s' English list (33 words, from Lucene) **plus** extra function words
   it misses: question words (`how`, `what`, `where`), pronouns (`we`, `our`, `them`),
   auxiliaries/modals (`do`, `can`, `should`), prepositions (`from`, `between`, `via`) and
   quantifiers (`most`, `every`). Only grammatical words, never topical ones.
4. Snowball/Porter2 stemming, so word forms match: `stories → stori`, `agents → agent`,
   `modeling → model`

Example: `"RAG agents for DPO and LoRA fine-tuning"` → `rag, agent, dpo, lora, fine, tune`.
Acronyms survive lowercasing because the query is lowercased the same way.

**Tokenizer versioning.** `TOKENIZER_VERSION` (currently `v2`) is saved with the index. Loading
an index built with a different version fails with a "rebuild" message, so query and document
tokens can never silently disagree. `run_system.py` records the version in each run's metadata.

**Why v2.** With bm25s' default list, `we` (in 17,529 of 17,933 papers), `how`, `do`, `what`
and `from` survived as query terms, so ~95-99% of the corpus "matched" every eval question and
`n_matched` was meaningless; `LLMs` never matched `LLM`. v2 dropped e.g. q2 from 17,766 to
11,336 matched papers and raised `llm` coverage from 8,771 (`llms` only) to 10,767 papers.

### 3. Index
At build time, `bm25s` precomputes, for every stem, which papers contain it and how often,
plus each paper's length. That precomputation is why queries take milliseconds. The index
(~15 MB) and the ordered list of paper ids are saved in `data/index/bm25/`.

### 4. Scoring (BM25)
Each query term adds points to each paper that contains it; a paper's score is the sum:

```
score(paper) = Σ over query terms t of
    IDF(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · len / avg_len))
```

| Piece | Meaning | Effect |
|---|---|---|
| `IDF(t)` | how rare `t` is across the corpus | rare terms (`stori`) count far more than common ones (`model`) |
| `tf` with `k1 = 1.5` | how often `t` appears in the paper | more mentions help, with diminishing returns |
| `b = 0.75`, `len / avg_len` | length normalization | long abstracts don't win just by being long |

The search scores **every** paper (cheap at ~18k documents), then picks the top k with
`np.argpartition` (O(n)) and sorts only those k. Scoring everything also tells us how many
papers matched at all.

### 5. Tracing (`autoresearch/tracing.py`)
Every command writes JSON events to `runs/<run_id>.jsonl`. A v0 `search` event:

| Field | Meaning |
|---|---|
| `query` | raw query text |
| `query_tokens` | tokens BM25 actually matched on |
| `n_matched` / `n_docs` | papers with score > 0 (contain at least one query token) / corpus size |
| `term_doc_freq` | per token: how many papers contain it |
| `hits` | `paper_id`, `rank`, `score` for the top k |
| `latency_s` | search time |

`n_matched` and `term_doc_freq` show how broad a query is and which terms drive that. Example
for q4 (*"Where do hallucinations in LLMs come from? ..."*) with tokenizer v2: 14,821 of 17,933
papers match, with `term_doc_freq` = `hallucin 1025, llm 10767, come 196, train 6729,
data 5301, combat 30`. Broad terms like `llm` and `train` inflate `n_matched`; the selective
ones (`hallucin`, `combat`) drive the ranking. Later versions add planner decisions, verifier verdicts and costs to the same files.

## Evaluation

Evaluation is two separate steps: the **system** produces results, an independent **judge**
grades them. The judge never sees how results were produced, so the same judge grades every
version (v0 BM25, v1 hybrid, the later agent) and comparisons are apples to apples. It also
keeps the system from grading itself once it gets its own LLM verifier in v2.

```bash
python scripts/run_system.py --system bm25 -k 50       # 1. run the system on every eval question
python scripts/llm_judge.py eval/runs/<run_id>       # 2. grade the saved results
```

### Eval questions (`eval/queries.jsonl`)
One JSON object per line with a stable `id` (`q1`-`q5`) that results and judgments refer to.
To add a question, add a line.

| id | Question | What it stresses |
|---|---|---|
| q1 | How is structural similarity in stories captured? How is it defined and what embedding models excel at it? | concept vs. keywords; author's own field |
| q2 | How do we build personalized agentic systems for education? How do we evaluate them? | sparse topic in these venues |
| q3 | How do we build deep research agents? How do we evaluate every component of deep research agents? | crowded, hot topic: precision and ranking |
| q4 | Where do hallucinations in LLMs come from? Is it in the training data? How can we combat it? | three distinct aspects |
| q5 | What are the most effective post-training methods for building domain-specific agents? | "most effective" tempts the judge to grade quality, not relevance |
| q6 | How are LLMs used to generate political stories or narratives, and how is their capability evaluated? | narrow topic; generation + evaluation |
| q7 | How do we measure whether stories can be linked back to real individuals or leak private information, and what methods reduce this re-identification risk? | spans two areas (synthetic data, privacy/re-identification) |
| q8 | How can knowledge graphs be combined with LLMs to answer complex multi-hop questions? | synonyms (KG / knowledge graph / graph RAG) |
| q9 | What recent techniques improve knowledge distillation from large to small language models? | improvements vs. limitations |
| q10 | How reliable are LLMs as automatic evaluators of generated text | bias, human agreement and mitigation |
| q11 | When should LLM systems rely on long-context models versus retrieval-augmented generation or explicit memory mechanisms? | comparison across approaches + agent memory |

### Step 1: run the system (`scripts/run_system.py`)
Runs the chosen system (`--system bm25`; new versions register in the `SYSTEMS` table) on
every question and writes `eval/runs/<run_id>/`:

| File | Contents |
|---|---|
| `results.jsonl` | one line per (question, rank): `query_id, rank, paper_id, score` |
| `run_meta.json` | system, k, questions, corpus size, timestamp, and `per_query` diagnostics: result count, latency, and for BM25 `query_tokens`, `n_matched` (papers with score > 0), `n_docs`, `term_doc_freq` |

The results file records only what the system decided; it is a frozen record that can be
re-graded later (different judge model, revised rubric) without re-running retrieval.

### Step 2: LLM judge (`scripts/llm_judge.py`)
- **Model:** `gpt-5.6-luna` via the OpenAI **Responses API** with structured output
  (`responses.parse` into a Pydantic schema), reasoning effort `low`. Key read from
  `OPENAI_API_KEY` or `openai_secrect.txt` (git-ignored).
- **Input:** the question plus ONE paper: title, venue and full abstract (looked up in the
  corpus by `paper_id`).
- **Pointwise judging:** one call per paper, not all 50 in one prompt. Avoids position bias and
  "lost in the middle", and a paper's grade doesn't depend on what else was retrieved, so
  grades can be reused across systems. Calls run in parallel (16 at a time).
- **Output schema:** `reason` (1-2 sentences), `aspects_addressed` (which parts of the
  question it covers), `score` (1-5).

**Rubric (1-5)**

| Score | Meaning |
|---|---|
| 5 | Directly relevant: main contribution squarely addresses the question or one of its aspects; would definitely be cited |
| 4 | Highly relevant: substantially addresses an aspect, partly or as a major (not sole) focus |
| 3 | Somewhat relevant: related topic, or a method/dataset/finding that informs an answer; background or related work |
| 2 | Marginally relevant: same area or keywords, wouldn't meaningfully help |
| 1 | Not relevant: different topic or superficial overlap |

The prompt also tells the judge to grade **relevance, not quality**, that covering one aspect
well is enough, to use only the title and abstract, and that shared keywords are not enough.

**Files written next to the results** (named by judge model, so re-judging with another model
adds files instead of overwriting):

| File | Contents |
|---|---|
| `judgments_<model>.jsonl` | per retrieved paper: rank, system score, title, venue, url, judge score, aspects, reason |
| `metrics_<model>.json` | per question and total: metrics below, tokens, cost |

**Cache:** `eval/judgments_cache.jsonl`, append-only, keyed by (question **text**, paper,
model, prompt version). A paper already graded for a question is never paid for again, even
when a different system retrieves it. Keying on text means editing a question never reuses
stale grades; bumping `PROMPT_VERSION` invalidates grades when the rubric changes.

### Metrics (per question)
- `score_dist`: count of each grade 5..1; `mean_score`
- `n_relevant(>=4)`: papers graded 4 or 5
- `P@10`, `P@20`, `P@50`: share of the top 10/20/50 graded >= 4
- usage: calls, input / cached / output / reasoning tokens, cost in USD

**Cost** is computed from the Responses API `usage` of every call at $0.20 / 1M input,
$0.02 / 1M cached input, $1.20 / 1M output tokens (reasoning tokens bill as output). A full
v0 run is 550 calls (11 questions x 50 papers), about $0.15 if nothing is cached.

### Known limits of this evaluation
- Precision only: without knowing all relevant papers in the corpus, recall can't be measured
  yet. Next step: **pooling** (TREC-style) — union the top results of several systems (BM25,
  dense, HyDE dense), have the author label that pool, and measure recall and nDCG against it.
- The judge is an LLM. Its agreement with human labels (e.g. on q1) should be measured before
  its numbers are trusted.

## v0 results (baseline)

Run `20261122-162321-bm25-k50` · system `bm25` (tokenizer `v2`) · k = 50 · corpus 17,933 papers (ACL, EMNLP, NAACL, EACL; 2025-2026) · judge `gpt-5.6-luna` (reasoning `low`, prompt `v1`) · relevant = score >= 4.

| q | Mean | 5 | 4 | 3 | 2 | 1 | Relevant (>=4) | P@10 | P@20 | P@50 | Papers matched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | 2.54 | 0 | 8 | 19 | 15 | 8 | 8/50 | 0.30 | 0.25 | 0.16 | 16,831 |
| q2 | 4.22 | 15 | 31 | 4 | 0 | 0 | 46/50 | 1.00 | 1.00 | 0.92 | 11,336 |
| q3 | 4.80 | 40 | 10 | 0 | 0 | 0 | 50/50 | 1.00 | 1.00 | 1.00 | 11,614 |
| q4 | 3.66 | 10 | 27 | 2 | 8 | 3 | 37/50 | 0.70 | 0.70 | 0.74 | 14,821 |
| q5 | 4.38 | 25 | 20 | 4 | 1 | 0 | 45/50 | 0.90 | 0.90 | 0.90 | 15,783 |
| q6 | 3.34 | 0 | 24 | 19 | 7 | 0 | 24/50 | 0.60 | 0.45 | 0.48 | 16,626 |
| q7 | 3.32 | 9 | 24 | 1 | 6 | 10 | 33/50 | 1.00 | 0.85 | 0.66 | 13,520 |
| q8 | 4.52 | 32 | 13 | 4 | 1 | 0 | 45/50 | 1.00 | 1.00 | 0.90 | 14,992 |
| q9 | 4.22 | 30 | 7 | 7 | 6 | 0 | 37/50 | 0.70 | 0.70 | 0.74 | 17,363 |
| q10 | 3.36 | 11 | 13 | 12 | 11 | 3 | 24/50 | 0.60 | 0.55 | 0.48 | 16,141 |
| q11 | 4.10 | 7 | 42 | 0 | 1 | 0 | 49/50 | 1.00 | 1.00 | 0.98 | 17,664 |
| **avg** | **3.86** | 179 | 219 | 72 | 56 | 24 | 398/550 | **0.80** | **0.76** | **0.72** | |

Columns 5..1 are how many of the 50 retrieved papers got each grade; the avg row sums them and macro-averages the rest (each question weighs the same). Papers matched = papers with BM25 score > 0.

**Cost.** Judging this run made 429 new API calls (325,087 input, 54,478 output incl. 7,883 reasoning tokens) for $0.130; the rest came from the cache. All 679 judgments in the cache cost $0.207 in total. Retrieval itself: ~1-2 ms per question, no API cost.

**Observations**
- **q1 is the hardest by far** (mean 2.54, P@50 0.16, no paper graded 5): BM25 matches *stories* but not the concept of *structural similarity*. This is the clearest target for dense search in v1.
- **Five questions have P@10 = 1.0** (q2, q3, q7, q8, q11) and q5 is at 0.9. Precision at the top is saturated, so it cannot show further gains; v1 adds pooled **recall@k and nDCG@k** for that.
- **The tokenizer fix (v1 → v2 stopwords + `LLMs → LLM`) mattered:** on the original 5 questions P@10 rose for q1 (0.2 → 0.3) and q2 (0.6 → 1.0), and q2's matched papers fell from 17,766 to 11,336.
- **Papers matched stays high (11k-18k of 17,933)** because broad content words remain (`model`, `llm`, `train`, `evalu`); it is a query-breadth signal, not a quality metric.
- Precision says nothing about **relevant papers BM25 never retrieved** (recall). Pooling in v1 addresses this.
## Design decisions

| Decision | Why |
|---|---|
| `bm25s` over `rank_bm25` | much faster (sparse matrices) and saves the index to disk |
| Title counted twice | titles carry the most topical keywords |
| Stemming + extended stopwords + acronym plurals, BM25 only | lets word forms match; dense models have their own tokenizers and need raw text |
| Score all docs, then top-k | gives `n_matched` for free; fine at this corpus size |
| Deterministic doc order | identical index across rebuilds, so traces and evals are comparable |
| JSONL traces from day one | the eval harness and the future reflector both read them |

## Current corpus

| Source | Papers |
|---|---|
| ACL 2025 / 2026 | 3,086 / 4,459 |
| EMNLP 2025 / 2026 | 3,214 / 5,182 |
| NAACL 2025 | 1,193 |
| EACL 2026 | 799 |
| **Total** | **17,933** (9 without abstracts) |

ICLR and NeurIPS are pending OpenReview sign-in; after fetching them, re-run `build-index`.

## Limitations (what later versions address)

- **Matches words, not meaning.** It cannot tell "structural similarity of stories" from
  "story generation". → v1 adds dense embeddings (bge) fused with BM25 via RRF.
- **Depends on the user's exact wording.** Missing a synonym means missing the paper.
  → v2's planner writes keyword queries and HyDE abstracts for each aspect of the question.
- **No notion of relevance quality.** Scores rank papers but don't say whether any are actually
  relevant. → v2's LLM verifier.
- **Abstracts only.** Details in the full text are invisible.

## Files

```
autoresearch/
  __init__.py   paths (data/, data/index/, runs/)
  __main__.py   CLI: build-index, search
  corpus.py     load + dedupe + order papers; text for BM25
  bm25.py       tokenizer, BM25Index (build/save/load/search), SearchResult
  tracing.py    JSONL run traces
  llm.py        OpenAI Responses API wrapper: structured output, retries, token + cost accounting
scripts/
  fetch_proceedings.py   download proceedings into data/raw/
  run_system.py            run a system on the eval questions -> eval/runs/<run_id>/results.jsonl
  llm_judge.py           grade a saved run with the LLM judge -> judgments + metrics
eval/
  queries.jsonl          the eval questions
  judgments_cache.jsonl  every judgment made so far (reused across runs)
  runs/<run_id>/         results, run metadata, judgments, metrics
```
