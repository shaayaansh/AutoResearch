# AutoResearch v1 — Hybrid search: BM25 + dense embeddings, fused with RRF

v1 adds **dense retrieval** (bge-base embeddings) and fuses it with v0's BM25 using
**Reciprocal Rank Fusion**. Still **no LLM in the system** — only the evaluation judge uses one.
v1 also upgrades evaluation from precision-only to **pooled recall and nDCG**, because v0's
precision was already saturated on half the questions.

**Headline (11 questions, macro-average):** hybrid beats BM25 on every metric —
P@10 0.800 → **0.927**, R@50 0.564 → **0.615**, nDCG@10 0.792 → **0.889**, nDCG@50 0.807 → **0.858**,
MRR 0.909 → **1.000**. Dense alone is roughly on par with BM25 on average, but the two find
very different relevant papers, which is why fusing them works.

## Quick start

```bash
pip install -r requirements.txt                       # adds torch + sentence-transformers
python -m autoresearch build-index --dense            # embed all papers once (~10 min on Apple GPU)
python -m autoresearch search "..." --retriever hybrid -k 20      # bm25 | dense | hybrid (default)

python scripts/run_system.py --system dense  -k 50    # run a system on the eval questions
python scripts/run_system.py --system hybrid -k 50
python scripts/llm_judge.py eval/runs/<run_id>        # grade each run (cached judgments are free)
python scripts/compare_runs.py eval/runs/<bm25> eval/runs/<dense> eval/runs/<hybrid>
```

## How it works

```
                       ┌─► BM25 (v0, unchanged) ── top 100 ─┐
question (raw text) ───┤                                    ├─► RRF (k=60) ─► top 50
                       └─► bge query embedding ─► cosine ── top 100 ─┘
                                 vs. 17,933 × 768 matrix
```

### 1. Dense index (`autoresearch/dense.py`)
- **Model:** `BAAI/bge-base-en-v1.5` (110M parameters, 768-dim embeddings, 512-token limit).
- **Document text:** `"{title}. {abstract}"`, raw — no stemming, no stopword removal, title
  once (BM25's title doubling is a keyword trick; an embedding model reads the text as a whole).
  The longest abstracts are ~250 words (99th percentile), so practically nothing is truncated.
- **Query text:** the raw question with bge's retrieval prefix
  `"Represent this sentence for searching relevant passages: "`. bge is *asymmetric*: queries get
  the instruction, documents don't.
- **Similarity:** embeddings are L2-normalized, so cosine similarity = dot product. Search is one
  matrix-vector product over all papers (exact, not approximate), then `argpartition` for top-k.
- **Storage:** a plain numpy matrix, `data/index/dense/embeddings.npy` (17,933 × 768 float32,
  ~55 MB) plus `paper_ids.json` and `meta.json` (model, dim, text version, device, build time).
  No vector database: at this size exact search takes milliseconds.
- **Device:** picked automatically — CUDA, then Apple MPS, then CPU (`--device` overrides). On
  CUDA the model runs in fp16 (~2× faster). Built on Apple MPS in **596 s**.
- **Versioning:** `TEXT_VERSION` is stored in the index; if `dense_text()` changes, loading a
  stale index fails with a rebuild message (same safeguard as BM25's `TOKENIZER_VERSION`).
- **Warm-up:** the model is loaded when the system starts, not on the first query, so query
  latency isn't polluted by model loading.

### 2. Fusion (`autoresearch/hybrid.py`)
Reciprocal Rank Fusion (Cormack et al., 2009):

```
RRF(paper) = Σ over retrievers r   1 / (k + rank_r(paper))        k = 60
```

- Each retriever contributes its **top 100** (`depth`), so a paper ranked ~70 by both can still
  make the fused top 50. A paper missing from one list gets nothing from that retriever.
- **Ranks only, no scores:** BM25 scores (~0-15) and cosines (~0.3-0.9) live on different scales;
  RRF never has to normalize them.
- **k = 60** flattens the top ranks, so *agreement* between retrievers beats one retriever's #1:
  a paper ranked 30th by both (2/90 = 0.0222) outranks one ranked 1st by only one (1/61 = 0.0164).
- Ties break on the best single rank, then paper id, so output is deterministic.
- Each fused hit records its rank in each retriever (`component_ranks`, e.g.
  `{"bm25": 3, "dense": 41}`), saved in `results.jsonl` for analysis.
- Per-question diagnostics in `run_meta.json`: BM25 tokens and `n_matched`, `overlap_at_depth`
  (papers in both top-100 lists), and how many fused results came from both / BM25 only / dense only.

### 3. System registry (`autoresearch/systems.py`)
`bm25`, `dense` and `hybrid` are registered in one place, used by both the `search` command and
`scripts/run_system.py`. Every version stays runnable, so any two can be compared with the same
evaluation code. BM25 inside hybrid is literally the v0 code path: for all 11 questions its query
tokens are identical to the v0 run, and all 330 hybrid results with a BM25 rank ≤ 50 sit at the
same BM25 rank as in the v0 run.

## Evaluation upgrade: pooled recall and nDCG (`scripts/compare_runs.py`)

v0's P@10 was 1.0 on five questions, so precision could no longer show improvement, and it says
nothing about relevant papers a system *missed*. v1 adds TREC-style **pooling**:

1. Run each system (`run_system.py`) and grade each run (`llm_judge.py`). Grades are cached per
   (question, paper), so a paper found by several systems is graded once.
2. `compare_runs.py` builds a **pool** per question: every paper any compared system returned in
   its top 50, with its judge grade. No LLM calls happen in this step.
3. Every system is scored against the same pool.

| Metric | Definition |
|---|---|
| P@10, P@50 | share of the top 10 / 50 graded relevant (score ≥ 4) |
| R@10, R@50 | share of the pool's relevant papers that appear in the top 10 / 50 |
| nDCG@10, nDCG@50 | graded ranking quality vs. the ideal ordering of the pool; gain = score − 1 (0-4), log2 discount; 1.0 = perfect |
| MRR | 1 / rank of the first relevant paper |
| unique_rel | relevant papers that only this system found (what it adds over the others) |

Recall is relative to the pool (the union of what these systems found), so it is a fair
comparison between them, not an absolute recall over the whole corpus. Output:
`eval/comparisons/<timestamp>-bm25-vs-dense-vs-hybrid/comparison.{md,json}`.

## v1 results

Runs: `bm25` = `20261122-162321-bm25-k50` (the v0 baseline), `dense` = `20261122-173000-dense-k50`,
`hybrid` = `20261122-173013-hybrid-k50`. Judge `gpt-5.6-luna` (reasoning `low`, prompt `v1`),
relevant = score ≥ 4, corpus 17,933 papers.

### Macro-average over 11 questions (unique_rel = total)

| system | P@10 | P@50 | R@10 | R@50 | nDCG@10 | nDCG@50 | MRR | unique_rel |
|---|---|---|---|---|---|---|---|---|
| bm25 (v0) | 0.800 | 0.724 | 0.132 | 0.564 | 0.792 | 0.807 | 0.909 | 137 |
| dense | 0.836 | 0.716 | 0.145 | 0.577 | 0.833 | 0.817 | 1.000 | 142 |
| **hybrid (v1)** | **0.927** | **0.765** | **0.165** | **0.615** | **0.889** | **0.858** | **1.000** | 34 |

### Per question: R@50 / nDCG@50

| q | pool relevant | bm25 | dense | hybrid |
|---|---|---|---|---|
| q1 story structure | 18 | 0.444 / 0.646 | **0.889 / 0.816** | 0.722 / 0.798 |
| q2 education agents | 77 | 0.597 / 0.905 | 0.558 / 0.904 | **0.610 / 0.936** |
| q3 deep research agents | 85 | 0.588 / **0.947** | 0.588 / 0.911 | 0.588 / 0.928 |
| q4 hallucination | 79 | 0.468 / 0.682 | **0.633 / 0.926** | 0.582 / 0.863 |
| q5 domain post-training | 73 | **0.616 / 0.895** | 0.575 / 0.859 | 0.603 / 0.860 |
| q6 political stories | 41 | 0.585 / 0.772 | 0.512 / 0.769 | **0.634 / 0.841** |
| q7 privacy / re-identification | 46 | **0.717 / 0.766** | 0.326 / 0.427 | 0.652 / 0.743 |
| q8 KG + multi-hop | 79 | 0.570 / 0.906 | 0.570 / 0.908 | **0.608 / 0.929** |
| q9 distillation | 59 | 0.627 / 0.813 | 0.559 / 0.754 | **0.644 / 0.847** |
| q10 LLM-as-judge | 56 | 0.429 / 0.637 | **0.643 / 0.844** | 0.589 / 0.804 |
| q11 long context vs RAG | 87 | 0.563 / **0.908** | 0.494 / 0.872 | 0.529 / 0.888 |

### Per run (each run's own judge metrics): mean score / P@10

| q | bm25 | dense | hybrid |
|---|---|---|---|
| q1 | 2.54 / 0.3 | 2.92 / 0.5 | 2.82 / 0.6 |
| q2 | 4.22 / 1.0 | 4.16 / 1.0 | 4.32 / 1.0 |
| q3 | 4.80 / 1.0 | 4.56 / 1.0 | 4.66 / 1.0 |
| q4 | 3.66 / 0.7 | 4.52 / 1.0 | 4.20 / 1.0 |
| q5 | 4.38 / 0.9 | 4.24 / 0.9 | 4.24 / 1.0 |
| q6 | 3.34 / 0.6 | 3.28 / 0.8 | 3.54 / 0.8 |
| q7 | 3.32 / 1.0 | 2.28 / 0.4 | 3.22 / 1.0 |
| q8 | 4.52 / 1.0 | 4.56 / 0.9 | 4.66 / 1.0 |
| q9 | 4.22 / 0.7 | 3.74 / 1.0 | 4.24 / 0.9 |
| q10 | 3.36 / 0.6 | 4.08 / 0.9 | 3.94 / 0.9 |
| q11 | 4.10 / 1.0 | 3.98 / 0.8 | 4.00 / 1.0 |

### How much BM25 and dense agree (hybrid run diagnostics)

| q | in both top-100 lists | fused top 50: from both | BM25 only | dense only |
|---|---|---|---|---|
| q1 | 32 | 32 | 8 | 10 |
| q2 | 26 | 26 | 12 | 12 |
| q3 | 36 | 36 | 8 | 6 |
| q4 | 31 | 31 | 11 | 8 |
| q5 | 24 | 24 | 11 | 15 |
| q6 | 27 | 27 | 14 | 9 |
| q7 | 20 | 20 | 13 | 17 |
| q8 | 47 | 45 | 1 | 4 |
| q9 | 27 | 27 | 11 | 12 |
| q10 | 15 | 15 | 17 | 18 |
| q11 | 17 | 17 | 17 | 16 |

### Cost and latency
- Judging: dense run 405 new calls, **$0.121**; hybrid run only **39** new calls, **$0.012** (its
  results are almost all papers already graded from the BM25 and dense runs). All 1,123
  judgments in the cache so far: $0.34.
- Retrieval: dense ~17 ms per question on Apple MPS (the first couple ~100-130 ms), hybrid ~18 ms,
  BM25 ~1-2 ms. Building the dense index: 596 s once on MPS.

### Observations
- **Hybrid is the best system overall** and the only one that is **never the worst** on any
  question (by R@50). It has the highest R@50 outright on q2, q6, q8 and q9, ties on q3, and
  otherwise sits between BM25 and dense.
- **BM25 and dense are complementary, not redundant.** Their top-100 lists share only 15-47
  papers per question, and each returns ~140 relevant papers that neither other system returned
  (unique_rel 137 vs 142).
  That disagreement is exactly what fusion exploits.
- **Dense wins on conceptual questions:** q1 (structural similarity of stories, R@50 0.44 → 0.89),
  q4 (hallucination, 0.47 → 0.63), q10 (LLM-as-judge, 0.43 → 0.64). These are phrased as ideas
  rather than in the field's exact terms.
- **BM25 wins when exact terms matter:** q7 (privacy). Dense drifts to the *"stories / linked"*
  part of the question and pulls in misinformation-narrative and user-profiling papers (dense
  P@10 0.4), while BM25's top results all match *re-identification / privacy / leak*
  (P@10 1.0). Hybrid recovers to P@10 1.0.
- **RRF averages, it doesn't pick the winner.** When one retriever is much better (q1, q4, q10
  dense; q7 BM25), hybrid lands between the two instead of matching the better one. A system
  that knows *which* retriever suits *which part* of the question would do better — that is
  what the planner in v2 is for.
- **Agreement drives the fused list.** Almost every paper in both top-100 lists made the fused
  top 50 (e.g. q8: 47 shared, 45 in the fused top 50), as RRF intends.
- **MRR 1.0 for dense and hybrid**: the first result is relevant for every question.
- **q1 is still the hardest** (only 18 relevant papers in the pool): its evaluation is the most
  sensitive to a few papers.

## Design decisions

| Decision | Why |
|---|---|
| bge-base-en-v1.5 | strong, standard open baseline; small enough to embed 18k papers on a laptop |
| Raw title + abstract for dense | embedding models need natural text; stemming/stopwords are a BM25 concern |
| Query instruction prefix | bge is trained asymmetrically; queries without it retrieve worse |
| Normalized vectors, exact numpy search | cosine = dot product; ~55 MB, ms latency; a vector DB or ANN index adds complexity and approximation for nothing at this size |
| Auto device (CUDA > MPS > CPU), fp16 on CUDA | runs anywhere, fast where possible |
| RRF, k = 60, depth 100 | rank-based (no score calibration), well-established default; depth lets papers ranked ~70 by both rise |
| Store `component_ranks` and agreement stats | lets us explain *why* hybrid ranked a paper where it did |
| One system registry | the CLI and eval run identical code; old versions stay comparable |
| Pooled evaluation | precision was saturated; recall/nDCG need a shared set of known relevant papers |
| Pool = union of compared systems' top 50, graded once | standard TREC practice; the cache makes adding a system nearly free |

## Limitations / next

- **One query for both retrievers.** The user's question goes to BM25 and dense verbatim. BM25
  wants keywords and synonyms, dense wants natural text or a hypothetical abstract (HyDE).
  → v2: the planner splits the question into aspects and writes a keyword query and a HyDE query
  per aspect.
- **Fusion is blind to which retriever is right** (see q1, q7). → per-aspect queries in v2 and,
  later, an LLM verifier that reranks.
- **Pooled recall is relative** to what these three systems found; relevant papers none of them
  retrieved are invisible. Every new system added to the pool makes recall estimates more complete.
- **Abstracts only**, as in v0.

## Files (new or changed in v1)

```
autoresearch/
  dense.py      bge embeddings, numpy index, device selection, query prefix
  hybrid.py     RRF fusion + agreement diagnostics
  systems.py    registry: bm25 | dense | hybrid
  __main__.py   build-index [--bm25|--dense], search --retriever
scripts/
  run_system.py     now runs any registered system; saves component_ranks + diagnostics
  compare_runs.py   pooled comparison: P/R/nDCG@10,50, MRR, unique_rel
data/index/dense/   embeddings.npy, paper_ids.json, meta.json
eval/comparisons/   comparison.md / comparison.json per comparison
```
