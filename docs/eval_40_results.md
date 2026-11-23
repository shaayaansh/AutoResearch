# Results on the 40-question eval set (all versions)

_Re-run 2026-11-23. This is the main comparison of all versions; the per-version docs
(`v0_README.md` … `v3_README.md`) keep their original 11-question results._

The eval set grew from 11 to **40 questions** (`eval/queries.jsonl`). The 29 new ones (q12–q40)
are **implicit multi-aspect** questions: one goal, with no "and"/"or" joining the parts, whose
sub-topics the planner has to infer. For example, *"How can hospitals safely use LLMs to draft
discharge summaries?"* needs clinical summarization, factual-error evaluation, clinician
evaluation and privacy. In the literature these are close to **implicit multi-step** questions
(StrategyQA) and **multi-faceted queries with subtopics** (IR diversity work). Each question has
a `style` tag:

- `explicit`: parts stated in the question (7);
- `implicit`: one goal whose parts the planner must infer (31: q5, q8, q12–q40);
- `focused`: a single topic (2).

The 29 new questions also have draft `reference_aspects` (expected sub-topics), which aren't
used yet.

Every system was re-run on all 40 questions with the code at tag `v3`:

| system | run |
|---|---|
| bm25 (v0) | `20261123-123347-bm25-k50` |
| hybrid (v1) | `20261123-123401-hybrid-k50` |
| hyde (v2.1) | `20261123-123415-hyde-k50` |
| agent (v3) | `20261123-123824-agent-k50` |

All use k = 50. The judge is `gpt-5.6-luna`, and relevant means a score ≥ 4. The pool is the
union of the 4 systems' top 50. The original v2 prompt isn't in the code anymore; v2.1 stands
in for v2.

## Macro-average over 40 questions

| system | P@10 | P@50 | R@10 | R@50 | nDCG@10 | nDCG@50 | MRR | unique_rel |
|---|---|---|---|---|---|---|---|---|
| bm25 (v0) | 0.645 | 0.535 | 0.085 | 0.337 | 0.682 | 0.652 | 0.822 | 262 |
| hybrid (v1) | 0.780 | 0.625 | 0.106 | 0.409 | 0.777 | 0.741 | 0.901 | 268 |
| hyde (v2.1) | 0.930 | 0.806 | 0.132 | 0.545 | 0.898 | 0.878 | 1.000 | 309 |
| agent (v3) | 0.965 | 0.864 | 0.138 | 0.592 | 0.926 | 0.915 | 1.000 | 465 |

`unique_rel` = relevant papers that only this system found.

**v3 is best on every metric** (tied with v2.1 on MRR, both 1.000), with the best nDCG@50 on 37 of 40 questions.
Each version improves on the last, with the largest step from v1 to v2.1, where the planner
was added:

| step | P@10 | R@50 | nDCG@50 |
|---|---|---|---|
| v0 → v1 | +13.5 | +7.2 | +8.9 |
| v1 → v2.1 | +15.0 | +13.6 | +13.7 |
| v2.1 → v3 | +3.5 | +4.7 | +3.7 |

(points)

## By question style

### Explicit (7 questions)
| system | P@10 | R@50 | nDCG@10 | nDCG@50 |
|---|---|---|---|---|
| bm25 (v0) | 0.800 | 0.401 | 0.742 | 0.766 |
| hybrid (v1) | 0.914 | 0.436 | 0.849 | 0.815 |
| hyde (v2.1) | 0.929 | 0.483 | 0.883 | 0.876 |
| agent (v3) | 0.943 | 0.483 | 0.907 | 0.891 |

### Implicit (31 questions)
| system | P@10 | R@50 | nDCG@10 | nDCG@50 |
|---|---|---|---|---|
| bm25 (v0) | 0.610 | 0.320 | 0.662 | 0.623 |
| hybrid (v1) | 0.742 | 0.400 | 0.750 | 0.720 |
| hyde (v2.1) | 0.932 | 0.556 | 0.898 | 0.873 |
| agent (v3) | 0.968 | 0.614 | 0.926 | 0.915 |

### Focused (2 questions)
| system | P@10 | R@50 | nDCG@10 | nDCG@50 |
|---|---|---|---|---|
| bm25 (v0) | 0.650 | 0.388 | 0.766 | 0.710 |
| hybrid (v1) | 0.900 | 0.455 | 0.928 | 0.807 |
| hyde (v2.1) | 0.900 | 0.590 | 0.956 | 0.948 |
| agent (v3) | 1.000 | 0.636 | 1.000 | 0.988 |

**Implicit questions are where the planner and the loop matter most.** Keyword and embedding
search on the raw question does much worse on implicit questions. The planner more than
closes that gap, and the agent loop widens the lead:

| R@50 | hybrid (v1) | hyde (v2.1) | agent (v3) |
|---|---|---|---|
| explicit | 0.436 | 0.483 | 0.483 |
| implicit | 0.400 | 0.556 | 0.614 |

On explicit questions, the v3 loop adds precision but no recall. On implicit questions, it adds
both.

## Per question: R@50 / nDCG@50 (best nDCG@50 in bold)

| q | style | pool rel | bm25 (v0) | hybrid (v1) | hyde (v2.1) | agent (v3) |
|---|---|---|---|---|---|---|
| q1 | explicit | 27 | 0.296 / 0.592 | 0.481 / 0.731 | **0.630 / 0.800** | 0.556 / 0.768 |
| q2 | explicit | 93 | 0.495 / 0.860 | 0.505 / 0.889 | 0.527 / 0.941 | **0.516 / 0.945** |
| q3 | explicit | 95 | 0.526 / 0.947 | 0.526 / 0.928 | 0.516 / 0.929 | **0.526 / 0.948** |
| q4 | explicit | 115 | 0.322 / 0.640 | 0.400 / 0.810 | 0.435 / 0.904 | **0.426 / 0.920** |
| q5 | implicit | 116 | 0.388 / 0.857 | 0.379 / 0.824 | 0.414 / 0.874 | **0.431 / 0.928** |
| q6 | explicit | 64 | 0.375 / 0.710 | 0.406 / 0.774 | 0.406 / 0.780 | **0.438 / 0.837** |
| q7 | explicit | 86 | 0.384 / 0.718 | 0.349 / 0.697 | 0.465 / 0.860 | **0.512 / 0.903** |
| q8 | implicit | 90 | 0.500 / 0.906 | 0.533 / 0.929 | 0.533 / 0.966 | **0.544 / 0.985** |
| q9 | focused | 82 | 0.451 / 0.810 | 0.463 / 0.844 | 0.585 / 0.962 | **0.610 / 0.992** |
| q10 | focused | 74 | 0.324 / 0.610 | 0.446 / 0.770 | 0.595 / 0.934 | **0.662 / 0.985** |
| q11 | explicit | 120 | 0.408 / 0.898 | 0.383 / 0.878 | **0.400 / 0.920** | 0.408 / 0.918 |
| q12 | implicit | 94 | 0.457 / 0.834 | 0.447 / 0.833 | 0.479 / 0.879 | **0.521 / 0.913** |
| q13 | implicit | 66 | 0.152 / 0.433 | 0.318 / 0.582 | 0.561 / 0.865 | **0.606 / 0.923** |
| q14 | implicit | 72 | 0.083 / 0.218 | 0.278 / 0.553 | 0.528 / 0.899 | **0.597 / 0.911** |
| q15 | implicit | 37 | 0.243 / 0.340 | 0.378 / 0.455 | 0.405 / 0.529 | **0.405 / 0.563** |
| q16 | implicit | 88 | 0.364 / 0.721 | 0.500 / 0.853 | 0.545 / 0.910 | **0.545 / 0.921** |
| q17 | implicit | 94 | 0.298 / 0.628 | 0.255 / 0.602 | 0.521 / 0.890 | **0.532 / 0.929** |
| q18 | implicit | 65 | 0.492 / 0.791 | 0.538 / 0.822 | 0.600 / 0.862 | **0.662 / 0.911** |
| q19 | implicit | 67 | 0.299 / 0.640 | 0.313 / 0.700 | **0.597 / 0.898** | 0.478 / 0.839 |
| q20 | implicit | 106 | 0.443 / 0.864 | 0.425 / 0.862 | 0.453 / 0.903 | **0.472 / 0.923** |
| q21 | implicit | 75 | 0.147 / 0.379 | 0.307 / 0.599 | 0.520 / 0.828 | **0.640 / 0.919** |
| q22 | implicit | 61 | 0.344 / 0.643 | 0.410 / 0.712 | 0.557 / 0.860 | **0.672 / 0.930** |
| q23 | implicit | 87 | 0.253 / 0.577 | 0.299 / 0.656 | 0.517 / 0.902 | **0.575 / 0.961** |
| q24 | implicit | 65 | 0.308 / 0.595 | 0.431 / 0.724 | 0.631 / 0.883 | **0.677 / 0.888** |
| q25 | implicit | 48 | 0.312 / 0.614 | 0.438 / 0.742 | 0.521 / 0.784 | **0.542 / 0.792** |
| q26 | implicit | 84 | 0.429 / 0.766 | 0.381 / 0.747 | 0.571 / 0.966 | **0.595 / 0.977** |
| q27 | implicit | 44 | 0.318 / 0.540 | 0.386 / 0.638 | 0.682 / 0.821 | **0.864 / 0.935** |
| q28 | implicit | 76 | 0.395 / 0.689 | 0.474 / 0.796 | 0.579 / 0.913 | **0.632 / 0.950** |
| q29 | implicit | 68 | 0.265 / 0.590 | 0.353 / 0.686 | 0.441 / 0.795 | **0.632 / 0.928** |
| q30 | implicit | 123 | 0.317 / 0.786 | 0.374 / 0.890 | 0.398 / 0.979 | **0.407 / 0.981** |
| q31 | implicit | 68 | 0.279 / 0.539 | 0.382 / 0.690 | 0.676 / 0.921 | **0.721 / 0.966** |
| q32 | implicit | 78 | 0.513 / 0.833 | 0.526 / 0.876 | 0.564 / 0.892 | **0.577 / 0.931** |
| q33 | implicit | 49 | 0.204 / 0.431 | 0.327 / 0.560 | 0.592 / 0.784 | **0.837 / 0.906** |
| q34 | implicit | 97 | 0.454 / 0.893 | 0.485 / 0.924 | 0.515 / 0.945 | **0.515 / 0.948** |
| q35 | implicit | 59 | 0.186 / 0.437 | 0.559 / 0.775 | 0.712 / 0.930 | **0.797 / 0.954** |
| q36 | implicit | 55 | 0.327 / 0.568 | 0.364 / 0.677 | 0.636 / 0.884 | **0.709 / 0.911** |
| q37 | implicit | 71 | 0.211 / 0.486 | 0.324 / 0.644 | 0.577 / 0.875 | **0.620 / 0.913** |
| q38 | implicit | 37 | 0.027 / 0.236 | 0.162 / 0.341 | 0.568 / 0.773 | **0.865 / 0.917** |
| q39 | implicit | 67 | 0.627 / 0.879 | 0.701 / 0.931 | 0.687 / 0.939 | **0.687 / 0.955** |
| q40 | implicit | 69 | 0.275 / 0.586 | 0.362 / 0.702 | 0.667 / 0.925 | **0.667 / 0.974** |

## What the v3 loop did

- **Rounds:** 1 round on 8 questions, 2 on 12, 3 on 20.
- **Stop reasons:**

  | stop reason | questions |
  |---|---|
  | round cap | 18 |
  | question target reached | 17 |
  | nothing left to try | 2 |
  | planner decided done | 2 |
  | no new relevant papers | 1 |

- **Coverage:** an average of 45.2
  verified-relevant papers per question; 17 of 40 reached the target of 50.
- **New aspects:** the reviser added 1
  new aspect in total across 40 questions. Verifiers' "missing" notes still rarely lead to a new
  aspect.
- **Cost:** verifiers $0.50, reviser $0.034; planner $0, since
  its plans came from the v2.1 run's cache (that run's new plans cost $0.022).
- **Latency:** 20.7 s per question on average.

### Verifier vs. judge agreement (agent top 50, 2,000 papers)

| tier in final list | judged relevant | judged not relevant | precision |
|---|---|---|---|
| verified relevant | 1538 | 79 | **0.951** |
| unverified or rejected (fill) | 190 | 193 | 0.496 |

## Observations

- **The planner under-decomposes implicit questions.** On the 29 new questions, v2.1 planned
  12 with a single aspect. The aspect counts were:

  | aspects in plan | questions |
  |---|---|
  | 1 | 12 |
  | 2 | 3 |
  | 3 | 7 |
  | 4 | 6 |
  | 5 | 1 |

  The draft reference aspects have 3–4 per question. The rule "a focused question gets 1
  aspect" (v2.1) seems to fire on implicit questions too. Some of these still do well, because
  v3 goes deeper on a productive single aspect (q27 R@50 0.682 → 0.864; q38 0.568 → 0.865).
  Others don't, such as q19 (figures in scientific papers, 1 aspect).
- **The loop's biggest gains are on implicit questions** with small pools:
  - q33 (LLMs vs. crowdworkers): 0.592 → 0.837;
  - q38 (training on LLM text): 0.568 → 0.865;
  - q29 (retrieval vs. parametric conflicts): 0.441 → 0.632;
  - q21 (compression): 0.520 → 0.640.
- **v3 has lower R@50 than v2.1 on 4 questions: q1, q2, q4, q19.**
  q1 and q19 are the same pattern: sparse topics where the verifier keeps around 20 papers, and
  the rest of the list is filled from the RRF merge of several query versions instead of v2.1's
  original list. q2 is the share-rule flaw documented in `v3_README.md`.
- **Verification holds up on new questions.** 95% of verified papers are judged relevant,
  against 50% of the fill, which matches the 11-question result (96%).

## Next
1. **v3.1 loop fixes:** keep going deeper while the question is below target (q2), and fill
   the list from the round-1 list (q1, q19).
2. **Planner decomposition for implicit questions:** revisit the "1 aspect" rule, and measure
   plan quality against `reference_aspects` (subtopic recall of the plan).
3. **Turn "missing" notes into new aspects** more reliably.

## Reproduce
```bash
for s in bm25 hybrid hyde agent; do python scripts/run_system.py --system $s -q; done
# judge each run, then:
python scripts/compare_runs.py eval/runs/20261123-123347-bm25-k50 eval/runs/20261123-123401-hybrid-k50 \
    eval/runs/20261123-123415-hyde-k50 eval/runs/20261123-123824-agent-k50
```
