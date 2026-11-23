# AutoResearch v3: verifiers + agent loop

> **Update: 40-question eval (2026-11-23).** Re-evaluated with all versions on the expanded eval set (q1–q40, 29 new implicit multi-aspect questions): P@10 0.965 · P@50 0.864 · R@50 0.592 · nDCG@50 0.915. v3 is best on every metric (MRR tied with v2.1) and on 37 of 40 questions by nDCG@50; 95% of verified papers are judged relevant. The results below are the original 11-question evaluation. Full comparison: [`eval_40_results.md`](eval_40_results.md).
<!-- ## Update: 40-question eval -->

v2.1 searches once: the planner writes queries, and whatever they return is the answer. v3 adds
the two things that make it an **agent**: it **checks** its own results, and it **acts on what it
learns**. One verifier per aspect reads the retrieved abstracts and keeps or drops each paper.
It also explains why the bad ones were retrieved and what is missing. The planner reads those
notes and rewrites the queries that aren't working. This repeats for at most 3 rounds.

**Headline (11 questions, macro-average, pooled with v1 hybrid and v2.1):** v3 is the best system
on every precision and ranking metric: P@10 **0.964** (v2.1: 0.936), nDCG@10 **0.945** (0.916),
nDCG@50 **0.921** (0.903). Recall is about the same (R@50 0.567 vs 0.562). It finds **135**
relevant papers that no other system found (v2.1: 100). Of the papers it verified as relevant,
**96.3%** were also judged relevant by the independent eval judge. The loop cost **$0.19** for
all 11 questions.

```
    question ─► planner ─► aspects
                  ▲            │
                  │            ▼   per aspect, in parallel
                  │      search (BM25 + HyDE dense, RRF), or go deeper ─► next unverified candidates
                  │            │
                  │            ▼
                  │      one verifier per aspect (LLM, batches of 10) ─► keep/drop + failure notes
                  │            │
                  └─ reviser ◄─┘  weak aspects get new queries; productive ones go deeper with the
                       │          same query; aspects that reached their share stay frozen
                       ▼  stop: question has 50 relevant / 0 new relevant / planner done / round cap
               final list = verified-relevant first (ranked by RRF), then unverified, then rejected
```

## Design

### Workflow vs. agent: code runs the loop, LLMs make judgments
The orchestrator (`AgentSearcher._run` in `autoresearch/agent.py`) is plain Python. It decides
which aspects to search, launches the verifiers, counts coverage, checks the stop rules and
enforces the constraints. The LLM is called for exactly three judgments: **planning** (round 0),
**verifying** (every round) and **revising** (between rounds). This makes every round
inspectable and every stop rule testable with a fake LLM, so the loop can't run away or quietly
break its own rules.

### 1. One verifier per aspect, in parallel (`autoresearch/verifier.py`)
- The orchestrator creates **one verifier for each aspect** of the question. A 3-aspect question
  gets 3 verifiers. They run concurrently (`asyncio.gather`), and a shared semaphore caps
  in-flight API calls at 16.
- Each verifier reads its aspect's top **30 unverified candidates** in **batches of 10**
  papers per LLM call (`gpt-5.6-luna`, low reasoning). The 3 batches also run in parallel.
- A paper is relevant only if (1) its main topic substantially addresses *this aspect*, and
  (2) it is about the question's **core subject**. This rule keeps out the v2 failure mode:
  "general RLHF" is not relevant to "post-training for domain-specific agents".
- Output (structured, Pydantic): for each paper, `relevant` and a reason of up to 15 words.
- Verdicts are cached per (question, aspect, paper, model, version). A paper is never checked
  twice for the same aspect, and reruns are free and deterministic.
- **The relevance standard never changes between rounds.** The loop cannot "succeed" by
  lowering the bar.

### 2. Batch-level notes: why papers failed, and what is missing (the key feature)
Because a verifier sees **10 papers at once**, it can describe *patterns*, not just
individual verdicts. With each batch it also returns:
- **`failure_pattern`**: why the irrelevant papers were retrieved. It uses four categories:
  **wrong meaning** (a keyword matched a different concept), **lost core subject** (right
  method, wrong subject), **too generic** and **off-topic**.
- **`missing`**: a sub-topic the aspect clearly needs that **no paper in the batch covers**.

These notes turn the verifier from a filter into a source of **feedback for the planner**. A
real example from q6 (*"How are LLMs used to generate political stories or narratives, and how
is their capability evaluated?"*), round 1:

> **why not:** Most irrelevant papers analyze, classify, explain, or annotate existing political
> narratives rather than generate them.
> **missing:** The batch lacks work on LLM-generated political campaign narratives, strategic
> political messaging, or targeted political persuasion.

The planner's response was to rewrite the BM25 query. It dropped the near-universal
`LLM large language model` terms that v2.1's prompt rule had failed to remove, and made
*generation* explicit:

```
round 1: LLM large language model political narrative story generation political communication
         campaign discourse news persuasion propaganda synthetic media storytelling
round 2: LLM generated political news campaign messaging strategic communication persuasive
         narratives propaganda storytelling ideological discourse synthetic media
```
q6's verified-relevant count went 6 → 12 → 19 over three rounds, and its R@50 rose 0.456 → 0.509.

### 3. The target is per question, not per aspect
The question needs **50 verified-relevant papers**, the same as the top 50 we judge. Each
aspect's **share** is 50 divided by the number of aspects. An earlier version used 8 per aspect,
and q5 (1 aspect, 9/30 relevant) stopped after one round even though its pool has 100+
relevant papers.

### 4. Each aspect's next move: covered, deeper, or weak
After every round, each aspect gets a status:

| status | when | next round |
|---|---|---|
| **COVERED** | reached its share | frozen, not searched again |
| **DEEPER** | below share, but ≥ 25% of the batch it just checked was relevant | **same query**; verify the next 30 results from the same list (no LLM call needed) |
| **WEAK** | below share and mostly irrelevant | the **reviser** writes a new BM25 query and HyDE abstract |

A working query is never thrown away just because the aspect hasn't reached its share yet. On
q10, one aspect yielded 25/30 relevant twice by going deeper with the same query.

### 5. The reviser (the planner's second job, `autoresearch/planner.py`)
- The reviser reads exactly what the orchestrator assembles for each aspect: its status,
  description, current queries, relevant count against its share, up to 5 relevant titles, the
  failure patterns and the missing notes.
- It returns structured output: `reasoning`, `done`, `updates` (new queries for weak aspects)
  and `new_aspects` (at most one per round).
- It is called when some aspect is WEAK, or when a verifier flagged a missing sub-topic and
  there is room for another aspect.
- The prompt has rules for each failure type: drop or disambiguate wrong-meaning keywords, put
  the core subject into both queries, replace generic words, and borrow terminology from the
  relevant examples.
- **Rules enforced in code, not just in the prompt:**
  - only WEAK aspects can be rewritten; other updates are logged as `ignored`;
  - at most 5 aspects in total and one new aspect per round;
  - duplicate new aspects are dropped;
  - an aspect's name and description never change (so its verdict cache stays valid).

### 6. Stop rules (checked in code before any LLM call)
1. The question reached 50 verified-relevant papers.
2. A round (after round 1) found **0 new** relevant papers.
3. The planner says `done` and no aspect is going deeper.
4. Round 3 finished (hard cap).

### 7. Final ranking: verified first, then fill
Each aspect's ranked lists (one per query version) are merged with RRF, then merged across
aspects with RRF, as in v2. The list is then sorted into three tiers, keeping RRF order within
each tier: **verified-relevant → unverified → rejected**. Rejected papers appear only if there
aren't enough others, which happened only on q5.

### 8. Round 1 reuses v2.1's plans
The planner's round-0 output comes from v2.1's plan cache, so round 1 retrieves exactly what
v2.1 retrieved. Every difference from v2.1 comes from verification and the loop.

### 9. Verifier ≠ eval judge
The system's verifier and the evaluation judge (`scripts/llm_judge.py`) use the same model but
different prompts and outputs:
- the verifier gives keep/drop per *aspect*, judged in batches;
- the judge gives a 1–5 grade per *question*, one paper at a time.

If they were the same, the system would be graded by its own judge. We report how often they
agree.

## Observability
- **Live narration.** For `verify`/`agent` systems, `run_system.py` prints every step: plan,
  per-aspect search/deeper/frozen, verifier results with failure notes, coverage against the
  target, reviser reasoning, rewritten queries, ignored updates and stop reason. Use `-q` to
  turn it off.
- **Full trace.** `eval/runs/<run>/trace.jsonl` has one line for every LLM exchange: planner,
  every verifier batch (exact input text, verdicts, notes) and every reviser call (the exact
  text the planner read, and its decision), with token usage and cache flags.
- **Viewer.** `scripts/show_trace.py <run> --query q6 [--round N] [--role verifier] [--full]`
  prints the conversation as readable rounds.
- **run_meta.json** has the per-round summary for each question: action, yield, coverage,
  revisions, stop reason, final tiers, and cost broken down by role.

## Results

Runs: `hybrid` (v1) `20261122-173013`, `hyde` (v2.1) `20261122-204750`, `agent` (v3)
`20261122-213020`, all k = 50. Judge `gpt-5.6-luna`, relevant = score ≥ 4. The pool is the union
of these 3 systems' top 50. bm25 and dense are not in this pool, so absolute recall values
differ from `v2.1_README.md`; compare rows within this table.

### Macro-average over 11 questions

| system | P@10 | P@50 | R@10 | R@50 | nDCG@10 | nDCG@50 | MRR | unique_rel |
|---|---|---|---|---|---|---|---|---|
| hybrid (v1) | 0.927 | 0.765 | 0.130 | 0.497 | 0.883 | 0.830 | 1.000 | 227 |
| hyde (v2.1) | 0.936 | 0.849 | 0.133 | 0.562 | 0.916 | 0.903 | 1.000 | 100 |
| **agent (v3)** | **0.964** | **0.865** | **0.137** | **0.567** | **0.945** | **0.921** | 1.000 | 135 |

`unique_rel` = relevant papers only this system found. (Hybrid's 227 is high because it shares
little with the two planner-based systems.)

### Per question: R@50 / nDCG@50

| q | pool rel | hybrid | hyde v2.1 | agent v3 |
|---|---|---|---|---|
| q1 | 26 | 0.500 / 0.757 | **0.654 / 0.829** | 0.538 / 0.789 |
| q2 | 78 | 0.603 / 0.900 | **0.628** / 0.952 | 0.615 / **0.956** |
| q3 | 90 | **0.556** / 0.928 | 0.544 / 0.929 | **0.556 / 0.948** |
| q4 | 106 | 0.434 / 0.810 | **0.472** / 0.904 | 0.462 / **0.921** |
| q5 | 103 | 0.427 / 0.824 | 0.466 / 0.874 | **0.485 / 0.928** |
| q6 | 57 | 0.456 / 0.774 | 0.456 / 0.780 | **0.509 / 0.836** |
| q7 | 68 | 0.441 / 0.700 | **0.588 / 0.864** | 0.574 / 0.857 |
| q8 | 83 | 0.578 / 0.929 | 0.578 / 0.966 | **0.590 / 0.985** |
| q9 | 71 | 0.535 / 0.844 | 0.676 / 0.962 | **0.704 / 0.992** |
| q10 | 68 | 0.485 / 0.770 | 0.647 / 0.934 | **0.721 / 0.985** |
| q11 | 102 | 0.451 / 0.893 | 0.471 / **0.937** | **0.480** / 0.934 |

v3 has the best nDCG@50 on 8 of 11 questions and the best R@50 on 7 of 11 (tied on q3).

### What the loop did, per question

Notation: `S` = searched (new or rewritten query), `D` = went deeper with the same query;
`x/30` = relevant out of checked.

| q | aspects | rounds | stop | verified relevant | rounds detail |
|---|---|---|---|---|---|
| q1 | 2 | 3 | round cap | 20 | r1 a1S 5, a2S 9 · r2 a1S 3, a2D 4 · r3 a1S 3, a2S 0 |
| q2 | 2 | 2 | no usable revisions | 38 | r1 a1S 16, a2S 19 · r2 a1D 9, a2D 8 |
| q3 | 2 | 2 | target reached | 56 | r1 a1S 19, a2S 19 · r2 a1D 19, a2D 5 |
| q4 | 3 | 2 | target reached | 66 | r1 a1S 15, a2S 7, a3S 25 · r2 a1D 16, a2S 9 |
| q5 | 1 | 3 | round cap | 24 | r1 a1S 9 · r2 a1D 8 · r3 a1D 7 |
| q6 | 2 | 3 | round cap | 19 | r1 a1S 4, a2S 5 · r2 a1S 3, a2S 6 · r3 a1S 5, a2S 6 |
| q7 | 2 | 3 | round cap | 38 | r1 a1S 11, a2S 20 · r2 a1D 5, a2D 4 · r3 a1S 0, a2S 4 |
| q8 | 3 | 2 | planner done | 44 | r1 a1S 23, a2S 26, a3S 11 · r2 a3D 6 |
| q9 | 4 | 1 | target reached | 67 | r1 29, 21, 29, 28 |
| q10 | 1 | 2 | target reached | 50 | r1 a1S 25 · r2 a1D 25 |
| q11 | 4 | 1 | target reached | 88 | r1 24, 30, 30, 4 |

Five questions reached the target of 50; the rich topics (q9, q11) did so in round 1.

### Verifier vs. judge agreement (top 50 of the agent run)

| tier in final list | judged relevant | judged not relevant | judge precision |
|---|---|---|---|
| verified relevant | 417 | 16 | **0.963** |
| unverified or rejected (fill) | 59 | 58 | 0.504 |

When the verifier keeps a paper, the independent judge agrees 96% of the time. The unverified
fill is only 50% relevant, which is why "verified first" raises precision.

### Cost and latency
- Verifiers: **$0.179** (148 LLM calls in total, 441k input + 84k output tokens). Reviser:
  **$0.011**. Planner: $0 (round 1 reused v2.1's cached plans). q5's verdicts were cached from
  the earlier q5 test runs, which cost about $0.01.
- Latency: **~21 s per question** on average (8 s for one round, up to 37 s for three),
  while v2.1 makes no verifier calls at all. Nearly all of v3's time goes to the LLM
  verifier calls.

## Observations

- **Precision and ranking improve almost everywhere.** P@10 +2.8 points and nDCG@50 +1.8 points
  over v2.1. The biggest gains are on q5 (nDCG@50 0.874 → 0.928), q10 (R@50 0.647 → 0.721) and
  q6 (R@50 0.456 → 0.509).
- **Going deeper is the most effective move.** When a query works, verifying further down its
  list keeps finding papers (q10: 25/30 twice; q3: 19/30 twice; q5: 9 → 8 → 7).
- **Rewrites help on hard topics, slowly.** q6 gained 3–6 relevant papers per rewrite, and the
  rewrites removed the generic `LLM large language model` terms that v2.1's prompt rule never
  got rid of. On q7, round 3's rewrite of a1 found 0.
- **The reviser never added a new aspect** in any question, even though verifiers flagged
  missing sub-topics often. That half of the feedback isn't turned into action yet.
- **q1 regressed** (R@50 0.654 → 0.538). It's a sparse topic (pool of 26): the verifier kept
  only 20 papers, and the other 30 slots were filled from RRF over *three* query versions per
  aspect, which ranks some of v2.1's original papers lower. Likely fix: fill from the round-1
  (v2.1) list, not the merged list.
- **A flaw in the share rule (q2, and partly q7).** Each of q2's aspects reached its share (25
  and 27), but many papers were relevant to both aspects, so the question had only 38 unique
  papers. Both aspects were then marked COVERED and the loop stopped with "no usable revisions",
  even though their yield was still about 30%. Fix: while the question is below its target,
  any productive aspect should keep going deeper, whatever its own count.
- **The verifier's false negatives aren't measured.** The 96% agreement covers only papers the
  verifier *kept*. Rejected papers rarely reach the top 50, so they are rarely judged.
  Measuring this would need judging a sample of rejected papers.

## Next steps
1. Fix the share rule (deeper whenever the question is below target) and the q1 fill order.
2. Turn `missing` notes into new aspects more reliably (prompt, or a code rule when the same gap
   is reported in two or more batches).
3. Judge a sample of verifier-rejected papers to measure verifier recall.
4. Move from finding papers to **synthesis**: summarize what has been done per aspect and
   suggest open research questions, using the verified set as the evidence base.

## Reproduce

```bash
python scripts/run_system.py --system agent                      # live narration; -q to silence
python scripts/llm_judge.py eval/runs/<agent-run-id>
python scripts/compare_runs.py eval/runs/20261122-173013-hybrid-k50 \
    eval/runs/20261122-204750-hyde-k50 eval/runs/<agent-run-id>
python scripts/show_trace.py eval/runs/<agent-run-id> --query q6  # read the agent's conversation
```

New files: `autoresearch/verifier.py`, `autoresearch/agent.py`, `scripts/show_trace.py`.
Changed: `autoresearch/planner.py` (reviser), `autoresearch/systems.py` (`verify`, `agent`),
`scripts/run_system.py` (narration, trace), `autoresearch/__main__.py`.
Caches to commit: `data/cache/verifications.jsonl`, `data/cache/revisions.jsonl`.
