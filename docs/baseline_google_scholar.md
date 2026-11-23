# External baseline: Google Scholar (top 10)

How does the system compare with what a researcher gets by typing the same question into
Google Scholar? Every system's **top 10** is graded by the same LLM judge with the same rubric.
No pooling, no recall: each system is scored only on its own results.

**Headline:** at top 10, the system is **on par with Google Scholar**: slightly higher precision
(P@10 hybrid 0.927, HyDE 0.909 vs. Google Scholar 0.882), slightly lower graded ranking
(nDCG@10 0.863 / 0.874 vs. 0.880), while searching a venue-specific corpus of 17,933 papers.
It clearly wins on the concept-level question where keyword matching fails
(q1: P@10 0.6 vs. 0.1). Differences elsewhere are small (≈5 relevant papers out of 110).

## Method

### 1. Collecting Google Scholar results (manual)
Google Scholar has no official API and blocks automated querying, so results were collected by
hand in the web interface:
- the **exact question text** from `eval/queries.jsonl`, unmodified (same input as our systems);
- filters: **Since 2025**, sorted by relevance (Google Scholar's defaults otherwise);
- the **top 10** results per question, all 11 questions (110 results);
- stored in `eval/manual/google_scholar/q<N>.jsonl`: rank, title, Google Scholar's author/venue
  line, PDF host, citation count, filters used.

Other options were tried first and dropped:
- **ACL Anthology search** is Google Custom Search; its JSON API has been closed to new
  customers since January 2026, and scraping Google violates its terms.
- **Semantic Scholar API**: anonymous access was rate-limited (HTTP 429 on every request);
  an API key would have been needed.

### 2. Abstracts (`scripts/fetch_gs_abstracts.py`)
The judge grades title + abstract, so every result needs its abstract:

| Source | Results | How |
|---|---|---|
| Our corpus | 6 | title matches a corpus paper: corpus abstract, corpus id (existing judgments reused) |
| OpenAlex API | 80 | title search; accepted only if normalized title similarity ≥ 0.92 (or prefix match for titles Google Scholar truncated with "…"); abstract rebuilt from OpenAlex's inverted index |
| Pasted by hand | 24 | publisher-restricted abstracts (Elsevier, Springer, ...) or titles not in OpenAlex; PDF artifacts cleaned (line-break hyphens, run-together words, page numbers, footnote markers, markdown links) |

The 0.92 threshold mattered: OpenAlex's closest candidates for the unmatched titles were all
*different* papers (similarity 0.36-0.83), and none were accepted.

Four abstracts are weaker and flagged with `abstract_note`: q10 #1 and q11 #6 are short
summaries (28 and 53 words), q3 #1 and q11 #10 are cut off mid-sentence. Pasted abstracts were
judged relevant at the same rate as OpenAlex ones (22/24 vs. 69/80), so abstract source did not
skew the grades.

### 3. Run and judging
- `scripts/import_google_scholar.py` builds a normal run folder
  (`eval/runs/20261122-200156-google_scholar-k10/`). Corpus papers keep their ids; the other 104
  get a stable id (`ext:` + hash of the normalized title) and carry title / abstract / venue inline.
- `scripts/llm_judge.py` grades them with the **same prompt, rubric and model** (`gpt-5.6-luna`)
  as every other paper. For papers outside the corpus it reads title/abstract/venue from the run
  file; for corpus papers the prompt is unchanged, so cached grades stay valid.
  Cost: 104 new judgments, **$0.032**.

### 4. Comparison (`scripts/compare_topk.py -k 10`)
All 10 Google Scholar results count, whatever their venue (this is what a researcher sees).
Our systems' runs (k = 50) are cut to their top 10.

| Metric | Definition |
|---|---|
| P@5, P@10 | share of the top 5 / 10 graded relevant (score ≥ 4) |
| mean@10 | average judge score (1-5) of the top 10 |
| MRR | 1 / rank of the first relevant paper |
| nDCG@10 | graded ranking quality vs. an absolute ideal (every slot a 5); gain = score − 1; uses only the run's own grades, no pool |

## Results

Comparison: `eval/comparisons/20261122-200358-top10-google_scholar-vs-bm25-vs-dense-vs-hybrid-vs-hyde/`.

### Macro-average over 11 questions

| system | P@5 | P@10 | mean@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| **hybrid (v1)** | **0.964** | **0.927** | 4.418 | 1.000 | 0.863 |
| **hyde (v2)** | 0.927 | 0.909 | **4.473** | 1.000 | 0.874 |
| Google Scholar | 0.909 | 0.882 | 4.455 | 1.000 | **0.880** |
| dense | 0.855 | 0.836 | 4.173 | 1.000 | 0.809 |
| bm25 (v0) | 0.782 | 0.800 | 4.055 | 0.909 | 0.774 |

### Per question: P@10 / mean score@10

| q | Google Scholar | bm25 | dense | hybrid | hyde |
|---|---|---|---|---|---|
| q1 story structure | **0.1 / 2.10** | 0.3 / 2.50 | 0.5 / 3.30 | **0.6 / 3.60** | 0.5 / 3.30 |
| q2 education agents | 1.0 / 4.80 | 1.0 / 4.30 | 1.0 / 4.50 | 1.0 / 4.60 | 1.0 / 4.60 |
| q3 deep research agents | 0.9 / 4.80 | 1.0 / 4.70 | 1.0 / 4.90 | 1.0 / 4.80 | 1.0 / 4.80 |
| q4 hallucination | 1.0 / 4.90 | 0.7 / 3.40 | 1.0 / 4.40 | 1.0 / 4.50 | 1.0 / 4.40 |
| q5 domain post-training | 0.9 / 4.50 | 0.9 / 4.50 | 0.9 / 4.20 | 1.0 / 4.60 | 0.9 / 4.60 |
| q6 political stories | 0.8 / 3.90 | 0.6 / 3.50 | 0.8 / 3.90 | 0.8 / 3.70 | 0.8 / 4.00 |
| q7 privacy / re-identification | 1.0 / 4.70 | 1.0 / 4.50 | 0.4 / 2.60 | 1.0 / 4.60 | 1.0 / 4.60 |
| q8 KG + multi-hop | 1.0 / 5.00 | 1.0 / 4.90 | 0.9 / 4.70 | 1.0 / 4.80 | 1.0 / 4.90 |
| q9 distillation | 1.0 / 4.80 | 0.7 / 4.30 | 1.0 / 4.90 | 0.9 / 4.70 | 1.0 / 5.00 |
| q10 LLM-as-judge | 1.0 / 5.00 | 0.6 / 3.80 | 0.9 / 4.60 | 0.9 / 4.50 | 1.0 / 5.00 |
| q11 long context vs RAG | 1.0 / 4.50 | 1.0 / 4.20 | 0.8 / 3.90 | 1.0 / 4.20 | 0.8 / 4.00 |

### Where Google Scholar's results come from

Only **6 of 110** (5.5%) are in our corpus (q4: 4, q10: 1, q11: 1). By PDF host: arXiv 32,
ACM 11, aclanthology.org 9 (of which 3 are other Anthology venues: COLING, LREC, a workshop),
Springer 8, IEEE 8, ScienceDirect 5, ICLR 5, MDPI 3, AAAI 2, others. Of the 104 results outside
our corpus, **91 (88%) were judged relevant**: many are highly cited surveys from other venues.

All 6 in-corpus Google Scholar papers were also retrieved by at least one of our systems in its
top 50, and all six were judged 4-5. v1 hybrid ranked the q4 papers at 2, 3, 9, 13 (Google Scholar:
4, 6, 7, 10) and the q10 paper at 2 (Google Scholar: 3). The q11 paper (Google Scholar rank 8)
was found only by dense (rank 22) and HyDE (rank 34), not by BM25 or hybrid.

## Observations

- **On par, not better.** With 11 questions, a 0.045 P@10 gap is ≈5 papers out of 110; neither
  side clearly wins on average. The fair claim is parity at top 10, with the system restricted to
  ~18k venue-specific papers and Google Scholar searching the whole literature.
- **The system wins where keywords fail.** On q1, Google Scholar matched "structural similarity"
  literally: papers on the SSIM image metric, protein language models, carbon-emission forecasting
  and SQL metrics (six results graded 1-2). Dense retrieval and HyDE find work on narrative
  structure instead (P@10 0.6 vs. 0.1).
- **Google Scholar wins on established topics.** Citation-based ranking puts landmark surveys first
  (q4: a hallucination survey cited 8,442 times; q10: highly cited LLM-as-judge studies), giving
  mean scores of 4.9-5.0. Our system uses no citation signal.
- **v2 does not beat v1 at top 10.** Hybrid has the best P@5/P@10; v2's gains are deeper in the
  list (P@50, R@50, see `v2_README.md`). Decomposition adds breadth, not top-10 precision.
- **Venue targeting is the real difference.** 94.5% of Google Scholar's top results fall outside
  ACL/EMNLP/NAACL/EACL; a researcher who needs that literature gets it from the system at equal
  precision.

## Caveats
- 11 questions × 10 results; no significance test yet (paired bootstrap / sign test is a next step).
- Google Scholar results are a snapshot (collected 2026-11-22, "Since 2025"); its ranking changes over time.
- One LLM judge; venue is shown to the judge for every paper (as for corpus papers), which could
  favor well-known venues slightly.
- Four results have weaker abstracts (see above).

## Reproduce

```bash
python scripts/fetch_gs_abstracts.py                       # fill abstracts (idempotent)
python scripts/import_google_scholar.py                    # -> eval/runs/<id>-google_scholar-k10/
python scripts/llm_judge.py eval/runs/<id>-google_scholar-k10
python scripts/compare_topk.py eval/runs/<gs> eval/runs/<bm25> eval/runs/<dense> eval/runs/<hybrid> eval/runs/<hyde> -k 10
```
