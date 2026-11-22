# AutoResearch v2 — LLM planner: aspect decomposition + HyDE

v2 is the first version with an **LLM inside the system**. A planner splits the research question
into aspects and writes, for each aspect, a keyword query for BM25 and a hypothetical abstract
(HyDE) for dense search. Each aspect runs its own hybrid search; the aspect lists are merged
with RRF.

**Headline (11 questions, macro-average, pool of 5 systems):** vs. v1 hybrid,
P@50 0.765 → **0.845**, R@50 0.414 → **0.473** (+14% of the known relevant papers),
nDCG@50 0.826 → **0.890**, nDCG@10 0.880 → **0.890**, and the most relevant papers no other
system found (**135**). P@10 is flat (0.927 → 0.909): the top 10 was already near its ceiling.
Planning costs **~$0.0013 per question**.

## Quick start

```bash
# see the plan and the results for one question
python -m autoresearch search "..." --retriever hyde -k 10

# evaluation: same plans for both systems (the second run reuses the cached plans)
python scripts/run_system.py --system aspects -k 50     # ablation: dense uses aspect descriptions
python scripts/run_system.py --system hyde -k 50        # v2: dense uses HyDE abstracts
python scripts/llm_judge.py eval/runs/<run_id>          # for each run
python scripts/compare_runs.py eval/runs/<bm25> eval/runs/<dense> eval/runs/<hybrid> eval/runs/<aspects> eval/runs/<hyde>
```

## How it works

```
question ─► planner (gpt-5.6-luna, 1 call) ─► 2-5 aspects, each with:
                                                 description · bm25_query · hyde_abstract
   for each aspect:
      BM25(bm25_query)  top 100 ─┐
                                  ├─► RRF (k=60) ─► aspect list (top 100)
      dense(probe)      top 100 ─┘
   RRF (k=60) across aspect lists ─► final top 50
```

### 1. Planner (`autoresearch/planner.py`)
One structured-output call per question (Responses API, `responses.parse` into a Pydantic
schema, reasoning effort `low`). The prompt tells the planner:
- the search runs over **titles and abstracts** of ACL/EMNLP/NAACL/EACL 2025-2026, with BM25 and
  dense retrievers;
- split the question into **2-5 aspects**: distinct sub-topics a complete literature review needs;
  a narrow question needs only 2, don't pad, minimal overlap, together cover every part;
- per aspect write:

| Field | What it is | Used by |
|---|---|---|
| `name` | 2-6 words | display, diagnostics |
| `description` | 1-2 plain sentences | dense search in the `aspects` ablation |
| `bm25_query` | 6-15 space-separated terms: exact technical terms, synonyms, acronyms **and** expansions; no quotes, operators, question words, filler, or near-universal words (model, LLM, method, ...) | BM25 |
| `hyde_abstract` | 100-150 word abstract of an ideal paper, in ACL style: problem, approach, evaluation, findings; real terminology; no invented benchmark/model names | dense search in `hyde` |

The planner chose 3-5 aspects (42 in total over 11 questions); HyDE abstracts came out at
120-141 words.

**Example plan, q7** (*"How do we measure whether stories can be linked back to real individuals
or leak private information, and what methods reduce this re-identification risk?"*):

| Aspect | BM25 query |
|---|---|
| Story Re-identification Measurement | re-identification de-identification anonymization linkage identity inference authorship attribution narrative text privacy attack reidentification |
| Private Information Leakage | privacy leakage sensitive attribute inference confidential information memorization membership inference disclosure risk personal narratives text privacy-preserving NLP |
| Privacy Risk Reduction | privacy-preserving text anonymization de-identification redaction pseudonymization differential privacy paraphrasing sanitization synthetic data narrative rewriting |
| Privacy Utility Evaluation | privacy utility tradeoff narrative quality fidelity coherence usefulness risk assessment threat model anonymization evaluation human study disclosure |

HyDE abstract for the first aspect (excerpt): *"We study how to measure re-identification risk in
narrative text that has been anonymized, transformed, or generated from records about real
people. We formalize an attacker who links a story to a candidate individual using names,
biographical details, writing style, event combinations, and external background knowledge..."*

### 2. HyDE (Gao et al., 2022)
A question and the abstract that answers it are different kinds of text, so their embeddings
sit apart. HyDE embeds a **hypothetical abstract** instead of the question: it reads like the
papers we want, so its embedding lands among them, and it brings in field vocabulary the user
didn't type. Its specifics can be wrong; only its "shape" matters.

- **Embedded as a document, without bge's query instruction** (`DenseIndex.embed_document`): the
  comparison is document-to-document, exactly how the corpus was embedded. The aspect
  description in the ablation is embedded as a query, with the instruction (`embed_query`).
- One abstract per aspect; no averaging with the question embedding (possible later experiments).

### 3. Per-aspect search and merging (`autoresearch/planned.py`)
- Each aspect: BM25 on its `bm25_query` (top 100) + dense on its probe (top 100), fused with RRF
  (k = 60), giving the aspect's ranked list (top 100). Same fusion as v1, per aspect.
- **Merging:** RRF again, over the aspect lists. Every aspect counts equally, so an aspect with
  many matching papers can't crowd out the others, and papers that several aspects retrieve rise.
- One persistent event loop per searcher: the async OpenAI client is bound to the loop it first
  ran on, so a fresh `asyncio.run()` per question would fail from the second question on.

### 4. Ablation design
`aspects` and `hyde` use **identical plans** (the same cached planner output) and identical BM25
queries. The only difference is the dense probe (description vs. HyDE abstract), so:
- **v1 hybrid → aspects** = what decomposition + planner keywords are worth;
- **aspects → hyde** = what HyDE adds on top.

### 5. What is stored
- **Plan cache** `data/cache/plans.jsonl` (committed): per (question, planner model, `PLANNER_VERSION`)
  the full plan and the call's token usage. Re-runs are deterministic and free; bumping
  `PLANNER_VERSION` forces new plans.
- **Per run** `run_meta.json`, per question: the full plan (every aspect's name, description,
  BM25 query, HyDE abstract), which probe dense search used (`dense_probe`), per aspect the BM25
  tokens, `bm25_n_matched`, BM25/dense overlap and its own ranked top-100 list (`retrieved`,
  with each paper's BM25 and dense ranks); `final_coverage_by_aspect` (how many of the final 50
  each aspect retrieved), `final_multi_aspect` (papers retrieved by 2+ aspects), planner
  latency, cache hit and token cost; plus `llm_usage_total` for the run.
- **Per result** in `results.jsonl`: `component_ranks` = the paper's rank in each aspect's list
  (e.g. `{"a1": 3, "a4": 12}`).
- The two evaluated runs below were made before `retrieved` and the `aspects`-mode HyDE text were
  added to `run_meta.json`; their complete plans are in the plan cache.

### 6. Evaluation is against the original question
The judge never sees the planner's aspects; it grades every paper against the user's question,
exactly as for v0 and v1. Aspects are system internals: grading against them would reward the
system for drifting into an off-topic aspect. It also keeps comparisons fair and the judgment
cache valid.

## v2 results

Runs: `bm25` `20261122-162321`, `dense` `20261122-173000`, `hybrid` `20261122-173013`,
`aspects` `20261122-180000`, `hyde` `20261122-180308` (all k = 50). Judge `gpt-5.6-luna`
(prompt `v1`), relevant = score ≥ 4. Pool = union of the top 50 of all 5 systems.

### Macro-average over 11 questions

| system | P@10 | P@50 | R@10 | R@50 | nDCG@10 | nDCG@50 | MRR | unique_rel |
|---|---|---|---|---|---|---|---|---|
| bm25 (v0) | 0.800 | 0.724 | 0.089 | 0.377 | 0.785 | 0.777 | 0.909 | 110 |
| dense | 0.836 | 0.716 | 0.099 | 0.394 | 0.824 | 0.787 | 1.000 | 86 |
| hybrid (v1) | **0.927** | 0.765 | **0.113** | 0.414 | 0.880 | 0.826 | 1.000 | 21 |
| aspects | 0.918 | 0.805 | 0.112 | 0.452 | 0.871 | 0.864 | 1.000 | 100 |
| **hyde (v2)** | 0.909 | **0.845** | 0.108 | **0.473** | **0.890** | **0.890** | 1.000 | **135** |

**Ablation** (macro-average deltas):

| step | ΔP@50 | ΔR@50 | ΔnDCG@50 |
|---|---|---|---|
| hybrid → aspects (decomposition + planner keywords) | +0.040 | +0.038 | +0.038 |
| aspects → hyde (HyDE probes) | +0.040 | +0.021 | +0.026 |
| **hybrid → hyde (all of v2)** | **+0.080** | **+0.059** | **+0.064** |

### Per question: R@50 / nDCG@50

| q | pool rel | bm25 | dense | hybrid | aspects | hyde |
|---|---|---|---|---|---|---|
| q1 story structure | 21 | 0.381 / 0.614 | 0.762 / 0.776 | 0.619 / 0.759 | **0.810** / 0.808 | **0.810 / 0.827** |
| q2 education agents | 105 | 0.438 / 0.857 | 0.410 / 0.856 | 0.448 / 0.886 | 0.457 / 0.900 | **0.467 / 0.922** |
| q3 deep research agents | 116 | **0.431 / 0.947** | **0.431** / 0.911 | **0.431** / 0.928 | 0.414 / 0.941 | 0.422 / 0.908 |
| q4 hallucination | 119 | 0.311 / 0.640 | **0.420** / 0.870 | 0.387 / 0.810 | 0.412 / **0.894** | **0.420** / 0.873 |
| q5 domain post-training | 120 | **0.375 / 0.857** | 0.350 / 0.822 | 0.367 / 0.824 | 0.308 / 0.803 | 0.325 / 0.819 |
| q6 political stories | 61 | 0.393 / 0.725 | 0.344 / 0.722 | 0.426 / 0.789 | **0.475 / 0.834** | 0.459 / 0.807 |
| q7 privacy / re-identification | 80 | 0.412 / 0.712 | 0.188 / 0.397 | 0.375 / 0.691 | 0.425 / 0.700 | **0.562 / 0.886** |
| q8 KG + multi-hop | 110 | 0.409 / 0.906 | 0.409 / 0.908 | 0.436 / 0.929 | 0.400 / 0.875 | **0.455 / 0.976** |
| q9 distillation | 96 | 0.385 / 0.810 | 0.344 / 0.751 | 0.396 / 0.844 | **0.490** / 0.950 | 0.479 / **0.956** |
| q10 LLM-as-judge | 103 | 0.233 / 0.610 | 0.350 / 0.808 | 0.320 / 0.770 | 0.427 / 0.937 | **0.456 / 0.964** |
| q11 long context vs RAG | 131 | **0.374 / 0.875** | 0.328 / 0.840 | 0.351 / 0.855 | 0.351 / 0.861 | 0.344 / 0.857 |

### Per run (each run's own judge metrics): P@10 / P@50 / mean score

| q | bm25 | hybrid | aspects | hyde |
|---|---|---|---|---|
| q1 | 0.3 / 0.16 / 2.54 | 0.6 / 0.26 / 2.82 | 0.6 / 0.34 / 3.02 | 0.5 / 0.34 / 3.06 |
| q2 | 1.0 / 0.92 / 4.22 | 1.0 / 0.94 / 4.32 | 1.0 / 0.96 / 4.44 | 1.0 / 0.98 / 4.46 |
| q3 | 1.0 / 1.00 / 4.80 | 1.0 / 1.00 / 4.66 | 1.0 / 0.96 / 4.70 | 1.0 / 0.98 / 4.56 |
| q4 | 0.7 / 0.74 / 3.66 | 1.0 / 0.92 / 4.20 | 1.0 / 0.98 / 4.58 | 1.0 / 1.00 / 4.54 |
| q5 | 0.9 / 0.90 / 4.38 | 1.0 / 0.88 / 4.24 | 1.0 / 0.74 / 4.06 | 0.9 / 0.78 / 4.18 |
| q6 | 0.6 / 0.48 / 3.34 | 0.8 / 0.52 / 3.54 | 0.9 / 0.58 / 3.66 | 0.8 / 0.56 / 3.56 |
| q7 | 1.0 / 0.66 / 3.32 | 1.0 / 0.60 / 3.22 | 0.7 / 0.68 / 3.44 | 1.0 / 0.90 / 4.14 |
| q8 | 1.0 / 0.90 / 4.52 | 1.0 / 0.96 / 4.66 | 1.0 / 0.88 / 4.50 | 1.0 / 1.00 / 4.90 |
| q9 | 0.7 / 0.74 / 4.22 | 0.9 / 0.76 / 4.24 | 0.9 / 0.94 / 4.78 | 1.0 / 0.92 / 4.76 |
| q10 | 0.6 / 0.48 / 3.36 | 0.9 / 0.66 / 3.94 | 1.0 / 0.88 / 4.68 | 1.0 / 0.94 / 4.80 |
| q11 | 1.0 / 0.98 / 4.10 | 1.0 / 0.92 / 4.00 | 1.0 / 0.92 / 4.00 | 0.8 / 0.90 / 4.04 |

### How the aspects shaped the final list (hyde run)

| q | aspects | papers each aspect contributed to the final 50 | final papers retrieved by 2+ aspects |
|---|---|---|---|
| q1 | 3 | 49 / 49 / 21 | 50 |
| q2 | 4 | 50 / 49 / 48 / 20 | 50 |
| q3 | 5 | 43 / 25 / 33 / 42 / 37 | 50 |
| q4 | 3 | 36 / 42 / 36 | 45 |
| q5 | 4 | 17 / 13 / 29 / 25 | **26** |
| q6 | 3 | 48 / 46 / 39 | 50 |
| q7 | 4 | 24 / 40 / 46 / 45 | 50 |
| q8 | 4 | 42 / 42 / 39 / 25 | 50 |
| q9 | 4 | 38 / 40 / 41 / 24 | 50 |
| q10 | 4 | 36 / 35 / 41 / 42 | 50 |
| q11 | 4 | 20 / 18 / 16 / 26 | **28** |

### Cost and latency
- **Planner:** 10 calls (q7's plan was already cached) · 5,479 input + 10,055 output tokens ·
  **$0.013 total, ~$0.0013 per question** · **7-11 s per question** to plan (uncached).
- **Retrieval with a cached plan:** ~0.1 s per question (3-5 aspects × BM25 + one embedding + 2 RRFs).
- **Judging:** `aspects` run 297 new calls ($0.090), `hyde` run 164 new calls ($0.050).
- Planning now dominates latency; retrieval itself is still fast.

### Observations
- **v2 improves every depth-50 metric.** Gains are largest in recall and ranking further down the
  list (P@50, R@50, nDCG@50), which is what decomposition is for: it finds relevant papers for
  parts of the question the raw query underweighted.
- **Earlier systems' recall fell (e.g. BM25 R@50 0.564 → 0.377) without anything changing in
  them**: v2 found many new relevant papers, so the pool grew (q10: 56 → 103 relevant papers,
  q4: 79 → 119). Pooled recall gets more accurate as systems are added.
- **Decomposition gives most of the recall gain (+0.038 of +0.059 R@50); HyDE adds as much
  precision again (+0.040 P@50 each).** The nDCG@50 gain splits roughly 60/40.
- **HyDE fixed v1's q7 failure.** The aspect descriptions barely helped (R@50 0.375 → 0.425);
  the HyDE abstracts did (→ 0.562, nDCG@50 0.691 → 0.886). A probe written like a privacy paper
  doesn't drift toward the word "stories" the way the question did.
- **Big wins on conceptual questions:** q10 LLM-as-judge (R@50 0.320 → 0.456), q9 distillation
  (0.396 → 0.479), q1 story structure (0.619 → 0.810).
- **Failure mode: aspects can drop the question's core requirement.** On q5 ("post-training for
  *domain-specific agents*") an aspect became generic RLHF/DPO, so its papers aren't about domain
  agents; on q11 (long context *versus* RAG *versus* memory) one aspect per approach finds papers
  about each approach, not the comparison. Both show low aspect agreement in the final list
  (26 and 28 papers retrieved by 2+ aspects vs. 45-50 elsewhere), and both lose to v1 on R@50.
- **Planner keywords are still too generic:** in the plans inspected (q3, q5, q7, q11), aspect
  BM25 queries match 8k-17k of 17,933 papers despite the prompt. BM25's IDF limits the damage, but tighter queries should help.
- **q3 is saturated** for all systems (~0.42-0.43 R@50, nDCG ~0.91-0.95).

## Design decisions

| Decision | Why |
|---|---|
| LLM only where judgment helps: planning | query understanding needs an LLM; retrieval and fusion stay deterministic and fast |
| One structured-output call per question | cheap (~$0.001), parseable, no free-text parsing |
| Planner chooses 2-5 aspects | narrow questions shouldn't be forced into many aspects |
| Separate query per retriever (keywords vs. HyDE) | BM25 needs exact terms and synonyms; dense needs document-like text |
| HyDE embedded as a document, no query prefix | document-to-document similarity, matching how the corpus was embedded |
| RRF per aspect, then across aspects | consistent with v1; every aspect counts equally; multi-aspect papers rise |
| Identical plans for `aspects` and `hyde` | clean ablation: only the dense probe differs |
| Plan cache keyed by (question, model, prompt version) | deterministic, free re-runs; version bump invalidates |
| Judge against the original question, never the aspects | the system's internals must not define the grading |
| Verifier deferred to v3 | one change per version, so each gain is attributable |

## Limitations / next

- **Aspects can lose the core requirement** (q5, q11). → v2.1: require every aspect to stay
  anchored to the question's core subject (and add a comparison aspect for comparison
  questions); add the original question's hybrid list as an "anchor" in the final RRF.
- **Generic BM25 keywords.** → stricter prompt rules / examples.
- **No relevance check.** Retrieved papers are never read. → v3: LLM verifier (separate prompt
  and ideally a different model from the eval judge, with agreement reported).
- **No aspect-coverage metric.** The judge's `aspects_addressed` is free text. → fixed reference
  aspects per eval question, written by hand, independent of the planner (needed for v4).
- **Fixed pipeline, no feedback.** → v4: the agent checks per-aspect coverage and expands weak
  aspects, up to 3 rounds.

## Files (new or changed in v2)

```
autoresearch/
  planner.py    Plan/Aspect schema, planner prompt, plan cache
  planned.py    per-aspect hybrid search, RRF across aspects, diagnostics; modes aspects | hyde
  dense.py      + embed_document (no prefix), search_vector
  systems.py    + aspects, hyde
  __main__.py   search shows the plan for planner systems
scripts/
  run_system.py totals the system's own LLM usage per run
data/cache/plans.jsonl   cached planner outputs (committed)
```
