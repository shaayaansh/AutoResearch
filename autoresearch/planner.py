"""Planner: one LLM call that turns a research question into a search plan.

For each ASPECT of the question the planner writes three things:
  description    plain-language statement of the aspect (used by the `aspects` ablation)
  bm25_query     keywords for BM25: exact terms, synonyms, acronyms + expansions
  hyde_abstract  a hypothetical abstract of the ideal paper for this aspect (HyDE, Gao et al. 2022),
                 embedded for dense search instead of the question

Plans are cached per (question, model, prompt version) in data/cache/plans.jsonl, so re-running
the system on the same question is deterministic and free. Bump PLANNER_VERSION when the prompt
or schema changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from . import DATA_DIR
from .llm import LLM, Usage

PLANNER_VERSION = "v2"  # v2: 1-5 aspects, aspects anchored to the core subject, few-shot examples
PLAN_CACHE = DATA_DIR / "cache" / "plans.jsonl"

# NOTE: the few-shot examples below are deliberately NOT eval questions (eval/queries.jsonl).
# Putting eval questions in the prompt would tune the planner to the test set (leakage).
PLANNER_INSTRUCTIONS = """\
You are an expert ML/NLP researcher planning a literature search. The search runs over the \
titles and abstracts of papers from ACL, EMNLP, NAACL and EACL (2025-2026), using two \
retrievers: BM25 keyword search and dense embedding search.

Given a research question, split it into its ASPECTS: the distinct sub-topics a complete \
literature review of the question must cover. Then write search queries for each aspect.

How many aspects:
- Use 1 to 5 aspects. A focused question that asks about one thing gets exactly 1 aspect. \
Only split when the question genuinely has separate parts. Never pad.
- Aspects should not overlap heavily, and together they must cover every part of the question.

Keep every aspect anchored to the question's core subject:
- First identify the question's core subject (e.g. "domain-specific agents", "low-resource NER", \
"political stories"). Every aspect must be about that core subject, not about a general topic. \
If the question asks about X for Y, every aspect is about X for Y, never X in general.
- If the question compares approaches (e.g. "A versus B", "when to use A or B"), each approach \
gets its own aspect tied to the core subject, AND add one aspect for work that directly \
compares them.

For each aspect write:

1. `name`: 2-6 words.

2. `description`: one or two plain sentences stating what papers for this aspect are about, \
including the core subject.

3. `bm25_query`: 6-15 space-separated search terms for a keyword engine that matches words, \
not meaning. It must include terms for the core subject. Use the field's exact technical \
terms, common synonyms, and both acronyms and their expansions (e.g. "RAG \
retrieval-augmented generation"). Prefer specific terms over broad ones. No quotes, no \
boolean operators, no question words or filler, and no words that appear in almost every NLP \
paper (model, language, LLM, method, approach, task, performance, results, evaluation) unless \
the aspect is specifically about them.

4. `hyde_abstract`: a 100-150 word abstract of an ideal paper on this aspect, written exactly \
as it would appear in ACL/EMNLP proceedings: a problem statement, the approach, how it is \
evaluated, and the findings. Use the field's real terminology. It is a search probe, not a \
real paper, so plausible specifics are fine, but do not invent named benchmarks or models.

=== Example 1: a focused question gets ONE aspect ===
Question: How do tokenizers affect the performance of multilingual language models?
Plan:
- name: Tokenization in multilingual models
  description: Papers that study how tokenizer design, such as vocabulary allocation across \
languages and subword segmentation, affects multilingual language model quality.
  bm25_query: tokenizer tokenization multilingual vocabulary subword segmentation fertility BPE \
SentencePiece cross-lingual low-resource languages
  hyde_abstract: Multilingual language models share a single subword vocabulary across many \
languages, yet how this vocabulary is allocated strongly shapes downstream quality. We analyze \
tokenizer fertility, the average number of subword tokens per word, across more than forty \
languages and relate it to model performance on classification, question answering and \
generation. Languages with non-Latin scripts and rich morphology receive substantially fewer \
vocabulary entries and are over-segmented, which increases sequence length and inference cost \
and correlates with lower accuracy. We compare BPE and unigram tokenizers trained with \
different language sampling rates and show that balanced vocabulary allocation narrows the \
gap for low-resource languages without hurting high-resource ones. Our results suggest that \
tokenizer design is an underexplored source of cross-lingual performance disparities.

=== Example 2: two aspects, BOTH anchored to the core subject ===
Question: How is chain-of-thought prompting used to improve code generation, and how is it \
evaluated?
(Core subject: chain-of-thought for code generation. Not "chain-of-thought in general".)
Plan:
- name: Chain-of-thought for code generation
  description: Papers that use chain-of-thought or step-by-step reasoning to improve the code \
that language models generate.
  bm25_query: chain-of-thought CoT step-by-step reasoning code generation program synthesis \
structured planning pseudocode coding
  hyde_abstract: Large language models can generate code from natural language, but they \
often fail on problems that require multi-step algorithmic reasoning. We study chain-of-thought \
prompting for code generation, in which the model first writes an explicit plan or pseudocode \
before producing the program. We compare free-form reasoning, structured plans and \
pseudocode-style intermediate steps across several open and proprietary models. Structured \
reasoning improves functional correctness on algorithmic problems, while free-form reasoning \
helps less and sometimes introduces errors that propagate into the code. Gains are largest for \
smaller models and for problems with longer solutions. We analyze failure cases and find that \
most remaining errors come from plans that are correct in outline but miss edge cases.
- name: Evaluating reasoning for code generation
  description: Papers that evaluate how chain-of-thought or step-by-step reasoning affects code \
generation quality, including benchmarks and metrics for reasoning-based code generation.
  bm25_query: code generation evaluation chain-of-thought reasoning benchmark pass@k functional \
correctness test cases execution program synthesis
  hyde_abstract: Reasoning-based prompting is widely used for code generation, but it is \
unclear how to evaluate whether the reasoning actually helps. We propose an evaluation \
protocol that measures both functional correctness, via execution against hidden test cases, \
and the faithfulness of intermediate reasoning to the final program. Applying it to several \
prompting strategies, we find that pass@k improvements often come with reasoning that does not \
match the generated code, and that standard benchmarks underestimate failures on edge cases. \
We release annotated reasoning traces and recommend reporting reasoning-code consistency \
alongside pass@k when evaluating chain-of-thought for code generation.

=== Example 3: a comparison question gets one aspect per approach PLUS a comparison aspect ===
Question: Should instruction tuning or in-context learning be used for low-resource named \
entity recognition?
(Core subject: low-resource NER. Every aspect stays on it.)
Plan:
- name: Instruction tuning for low-resource NER
  description: Papers that fine-tune or instruction-tune language models for named entity \
recognition when little labeled data is available.
  bm25_query: instruction tuning fine-tuning named entity recognition NER low-resource few-shot \
limited labeled data domain adaptation
  hyde_abstract: Named entity recognition in specialized domains and low-resource languages \
suffers from scarce annotated data. We study instruction tuning of language models for NER, \
casting entity extraction as instruction-following with natural language label descriptions. \
Training on a mixture of NER datasets with unified instructions lets the model generalize to \
new entity types from a handful of examples. Across biomedical, legal and multilingual \
benchmarks, instruction-tuned models outperform standard token-classification fine-tuning \
when fewer than one hundred labeled sentences are available. We analyze how instruction \
wording and label descriptions affect transfer to unseen entity types.
- name: In-context learning for low-resource NER
  description: Papers that use prompting or in-context examples, without fine-tuning, to \
perform named entity recognition with little labeled data.
  bm25_query: in-context learning ICL prompting demonstrations named entity recognition NER \
few-shot zero-shot low-resource
  hyde_abstract: In-context learning lets large language models perform named entity \
recognition from a few demonstrations, without updating parameters. We investigate how \
demonstration selection, output format and label definitions affect few-shot NER in \
low-resource settings. Retrieving demonstrations similar to the test sentence and asking for \
structured outputs both improve entity-level F1 substantially over random demonstrations. \
However, in-context learning remains sensitive to entity boundary errors and to rare entity \
types. We evaluate on several low-resource NER datasets and discuss when prompting is a \
practical alternative to annotation.
- name: Comparing fine-tuning and prompting for NER
  description: Papers that directly compare instruction tuning or fine-tuning with in-context \
learning for named entity recognition in low-resource settings.
  bm25_query: comparison fine-tuning in-context learning prompting named entity recognition NER \
low-resource few-shot trade-off cost
  hyde_abstract: Practitioners building NER systems with little labeled data must choose between \
fine-tuning smaller models and prompting large ones. We present a controlled comparison of \
instruction tuning, standard fine-tuning and in-context learning for low-resource named entity \
recognition, varying the number of labeled examples, the domain and the model size. \
Fine-tuned models overtake in-context learning once a few hundred labeled sentences are \
available, while prompting is competitive in the extreme few-shot regime and for new entity \
types. We also compare inference cost and annotation effort, and provide guidance on which \
approach to use under different data and compute budgets.
"""


class Aspect(BaseModel):
    name: str = Field(description="2-6 word name of the aspect")
    description: str = Field(description="One or two plain sentences describing the aspect")
    bm25_query: str = Field(description="6-15 space-separated keyword search terms")
    hyde_abstract: str = Field(description="100-150 word hypothetical abstract of an ideal paper")


class Plan(BaseModel):
    aspects: list[Aspect] = Field(description="1 to 5 aspects of the research question")


class Planner:
    def __init__(self, llm: LLM, cache_path: Path = PLAN_CACHE, use_cache: bool = True):
        self.llm = llm
        self.cache_path = cache_path
        self.use_cache = use_cache
        self._cache: dict[tuple, dict] = {}
        if use_cache and cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                r = json.loads(line)
                self._cache[(r["question"], r["model"], r["planner_version"])] = r

    async def plan(self, question: str) -> tuple[Plan, Usage, bool]:
        """Returns (plan, usage of this call, whether it came from the cache)."""
        key = (question, self.llm.model, PLANNER_VERSION)
        if self.use_cache and key in self._cache:
            return Plan.model_validate(self._cache[key]["plan"]), Usage(), True

        plan, usage = await self.llm.parse(instructions=PLANNER_INSTRUCTIONS,
                                           input=f"Research question:\n{question}", schema=Plan)
        if not plan.aspects:
            raise ValueError(f"Planner returned no aspects for: {question!r}")
        plan.aspects = plan.aspects[:5]  # enforce the upper bound even if the model overshoots
        record = {"question": question, "model": self.llm.model, "planner_version": PLANNER_VERSION,
                  "plan": plan.model_dump(), "usage": usage.as_dict()}
        self._cache[key] = record
        if self.use_cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return plan, usage, False
