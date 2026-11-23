"""v3: verifiers + the agent loop.

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

The orchestration is plain code (a workflow); the LLM is called only for judgment: planning,
verifying, revising. That makes every round inspectable and every stop condition testable.

`max_rounds=1` gives the "verifier only" ablation (v3a): same verification and re-ranking,
no loop. Comparing it with the full agent separates what verification buys from what the loop
buys.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

from .bm25 import BM25Index
from .corpus import Corpus
from .dense import DenseIndex
from .hybrid import DEPTH, RRF_K, rrf
from .llm import Usage
from .planner import Aspect, Planner, Reviser, format_revision_input
from .verifier import VerifierPool

TARGET_RELEVANT = 50  # verified-relevant papers wanted for the QUESTION (= the top-k we judge)
DEEPER_YIELD = 0.25   # an aspect whose last batch was >= 25% relevant keeps its query and goes deeper
MAX_ROUNDS = 3        # hard cap on search rounds
N_CANDIDATES = 30     # papers verified per aspect per round
MAX_ASPECTS = 5
N_GOOD_TITLES = 5     # relevant examples shown to the reviser per aspect

TIER_RELEVANT, TIER_UNVERIFIED, TIER_REJECTED = 0, 1, 2
TIER_NAMES = {TIER_RELEVANT: "relevant", TIER_UNVERIFIED: "unverified", TIER_REJECTED: "rejected"}


def _short(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _write(msg: str) -> None:
    try:  # play nicely with the progress bar in scripts/run_system.py
        from tqdm import tqdm
        tqdm.write(msg)
    except ImportError:
        print(msg, flush=True)


@dataclass
class AgentHit:
    paper_id: str
    score: float
    rank: int
    component_ranks: dict = field(default_factory=dict)  # aspect id -> rank in that aspect's list
    tier: str = "unverified"


@dataclass
class AspectState:
    id: str
    name: str
    description: str
    bm25_query: str
    hyde_abstract: str
    lists: list[list[str]] = field(default_factory=list)  # one ranked list per query version
    verdicts: dict[str, bool] = field(default_factory=dict)  # paper -> relevant (for this aspect)
    relevant: list[str] = field(default_factory=list)       # in the order they were found
    failure_patterns: list[str] = field(default_factory=list)  # latest round only
    missing: list[str] = field(default_factory=list)
    needs_search: bool = True     # new or rewritten queries: run the search
    go_deeper: bool = False       # same results list, verify the next candidates
    last_yield: float = 0.0       # share of the last round's checked papers that were relevant
    status: str = ""              # after each round: COVERED / DEEPER / WEAK
    added_in_round: int = 1
    query_history: list[dict] = field(default_factory=list)

    @property
    def n_relevant(self) -> int:
        return len(self.relevant)


class AgentSearcher:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, corpus: Corpus, planner: Planner,
                 verifiers: VerifierPool, reviser: Reviser | None, max_rounds: int = MAX_ROUNDS,
                 target: int = TARGET_RELEVANT, n_candidates: int = N_CANDIDATES,
                 depth: int = DEPTH, k: int = RRF_K, verbose: bool = False):
        self.verbose = verbose
        self.bm25, self.dense, self.corpus = bm25, dense, corpus
        self.planner, self.verifiers, self.reviser = planner, verifiers, reviser
        self.max_rounds = max_rounds if reviser is not None else 1
        self.target, self.n_candidates = target, n_candidates
        self.depth, self.k = depth, k
        self._loop = asyncio.new_event_loop()  # see PlannedSearcher: the async client is loop-bound

    # -- one aspect's retrieval: identical to v2.1 (planned.py, mode="hyde") ----------------------
    def _search_aspect(self, a: AspectState) -> dict:
        b = self.bm25.search(a.bm25_query, k=self.depth)
        d = self.dense.search_vector(self.dense.embed_document(a.hyde_abstract), k=self.depth)
        fused = rrf({"bm25": [h.paper_id for h in b.hits], "dense": [h.paper_id for h in d.hits]},
                    k=self.k)[: self.depth]
        ids = [h.paper_id for h in fused]
        a.lists.append(ids)
        a.query_history.append({"bm25_query": a.bm25_query, "hyde_abstract": a.hyde_abstract})
        a.needs_search = False
        return {"bm25_tokens": b.query_tokens, "bm25_n_matched": b.n_matched, "ids": ids}

    def _log(self, msg: str, t0: float | None = None) -> None:
        """Live narration of the loop (verbose=True). Everything printed here is also in diag."""
        if self.verbose:
            stamp = f"[{time.perf_counter() - t0:6.1f}s] " if t0 is not None else ""
            _write(f"  {stamp}{msg}")

    def _candidates(self, a: AspectState, ids: list[str]) -> list[dict]:
        """The top papers of the newest list that this aspect's verifier hasn't seen yet."""
        fresh = [pid for pid in ids if pid not in a.verdicts][: self.n_candidates]
        return [self.corpus.get(pid) for pid in fresh]

    async def _run(self, question: str) -> tuple[list[AspectState], list[dict], dict, str]:
        T = time.perf_counter()
        log = lambda m: self._log(m, T)  # noqa: E731
        if self.verbose:
            _write(f"\n▶ {_short(question, 150)}")
        usage = {"planner": Usage(), "verifier": Usage(), "reviser": Usage()}
        trace: list[dict] = []  # every LLM exchange, in order (written to trace.jsonl by run_system)
        log("planner: decomposing the question…")
        plan, u, plan_cached = await self.planner.plan(question)
        usage["planner"].add(u)
        trace.append({"round": 0, "role": "planner", "cached": plan_cached,
                      "input": f"Research question:\n{question}", "output": plan.model_dump(),
                      "usage": u.as_dict()})
        aspects = [AspectState(f"a{i}", x.name, x.description, x.bm25_query, x.hyde_abstract)
                   for i, x in enumerate(plan.aspects, start=1)]
        log(f"planner: {len(aspects)} aspect(s){' (cached plan)' if plan_cached else ''}")
        for a in aspects:
            log(f"   {a.id} {a.name} | bm25: {_short(a.bm25_query, 80)}")

        rounds, stop_reason = [], "round cap"
        ever_relevant: set[str] = set()
        for rnd in range(1, self.max_rounds + 1):
            t0 = time.perf_counter()
            log(f"── round {rnd}/{self.max_rounds} ──")
            # 1. each aspect: search (new/rewritten queries), go deeper (same results, next
            #    candidates), or stay frozen (covered its share, or nothing left to try)
            searched, work = {}, []
            for a in aspects:
                if a.needs_search:
                    s = self._search_aspect(a)
                    searched[a.id] = s
                    work.append((a, self._candidates(a, s["ids"])))
                    log(f"search   {a.id}: BM25 matched {s['bm25_n_matched']:,} docs; "
                        f"{len(work[-1][1])} new candidates to verify")
                elif a.go_deeper:
                    a.go_deeper = False
                    work.append((a, self._candidates(a, a.lists[-1])))
                    log(f"deeper   {a.id}: same query, next {len(work[-1][1])} candidates "
                        f"(last round {a.last_yield:.0%} relevant)")
                elif a.status == "WEAK":
                    log(f"idle     {a.id}: weak ({a.n_relevant} relevant) but the planner left it unchanged")
                else:
                    log(f"frozen   {a.id}: {a.n_relevant} relevant (covered its share), not searched")
            # 2. one verifier per active aspect, all in parallel
            log(f"verify   {len(work)} verifier(s) in parallel, "
                f"{sum(len(ps) for _, ps in work)} papers in batches of {self.verifiers.batch_size}…")
            vu = Usage()
            round_trace: list[dict] = []

            async def verify_one(a, ps):
                rep = await self.verifiers.verify_aspect(
                    question, {"id": a.id, "name": a.name, "description": a.description}, ps, vu,
                    round_trace)
                log(f"verified {a.id}: {rep.n_relevant}/{rep.n_checked} relevant"
                    f"{f', {rep.n_cached} cached' if rep.n_cached else ''}"
                    f"{f', {rep.n_errors} failed' if rep.n_errors else ''}")
                for fp in rep.failure_patterns[:2]:
                    log(f"           why not: {_short(fp)}")
                for m in rep.missing[:1]:
                    log(f"           missing: {_short(m)}")
                return rep

            reports = await asyncio.gather(*(verify_one(a, ps) for a, ps in work))
            usage["verifier"].add(vu)
            # batches finish in any order; store them grouped by aspect for readability
            trace.extend({"round": rnd, **t} for t in sorted(round_trace, key=lambda t: t["aspect_id"]))
            new_relevant = set()
            round_aspects = []
            for (a, papers), rep in zip(work, reports):
                n_rel_round = 0
                for p in papers:
                    v = self.verifiers.cached(question, a.description, p["id"])
                    if v is None:  # verification failed: stays unverified
                        continue
                    a.verdicts[p["id"]] = v["relevant"]
                    if v["relevant"]:
                        n_rel_round += 1
                        a.relevant.append(p["id"])
                        if p["id"] not in ever_relevant:
                            new_relevant.add(p["id"])
                a.last_yield = n_rel_round / rep.n_checked if rep.n_checked else 0.0
                a.failure_patterns, a.missing = rep.failure_patterns, rep.missing
                round_aspects.append({
                    "id": a.id, "name": a.name, "bm25_query": a.bm25_query,
                    "action": "search" if a.id in searched else "deeper",
                    "bm25_tokens": searched[a.id]["bm25_tokens"] if a.id in searched else None,
                    "bm25_n_matched": searched[a.id]["bm25_n_matched"] if a.id in searched else None,
                    "n_checked": rep.n_checked, "n_relevant_new": rep.n_relevant,
                    "yield": round(a.last_yield, 3), "n_relevant_total": a.n_relevant,
                    "n_cached": rep.n_cached, "n_errors": rep.n_errors,
                    "failure_patterns": rep.failure_patterns, "missing": rep.missing})
            ever_relevant |= new_relevant
            total_rel = len(ever_relevant)
            share = math.ceil(self.target / len(aspects))  # each aspect's part of the question target
            rec = {"round": rnd, "aspects": round_aspects, "n_new_relevant": len(new_relevant),
                   "n_relevant_total": total_rel, "target": self.target, "share_per_aspect": share,
                   "coverage": {a.id: a.n_relevant for a in aspects},
                   "latency_s": round(time.perf_counter() - t0, 3)}
            rounds.append(rec)
            log(f"coverage {total_rel}/{self.target} relevant for the question · "
                + ", ".join(f"{a.id}={a.n_relevant}" for a in aspects)
                + f" (share {share} each) · {len(new_relevant)} new this round")

            # 3. stop conditions checked in code before asking the LLM (cheap and predictable)
            if total_rel >= self.target:
                stop_reason = "question target reached"
                break
            if rnd > 1 and not new_relevant:
                stop_reason = "no new relevant papers"
                break
            if rnd == self.max_rounds:
                stop_reason = "round cap" if self.reviser else "verifier only (no loop)"
                break

            # 4. decide each aspect's next move: covered / deeper (query works) / weak (rewrite)
            status = {}
            for a in aspects:
                has_more = any(pid not in a.verdicts for pid in a.lists[-1])
                if a.n_relevant >= share:
                    status[a.id] = "COVERED"
                elif a.last_yield >= DEEPER_YIELD and has_more:
                    status[a.id] = "DEEPER"
                else:
                    status[a.id] = "WEAK"
            for a in aspects:
                a.status = status[a.id]
            weak = [a for a in aspects if status[a.id] == "WEAK"]
            deeper = [a for a in aspects if status[a.id] == "DEEPER"]
            for a in deeper:
                a.go_deeper = True
            # the reviser rewrites weak aspects and may add one for a missing sub-topic, so it is
            # also asked when nothing is weak but a verifier flagged a gap and there is room
            gaps = any(a.missing for a in aspects) and len(aspects) < MAX_ASPECTS
            if not weak and not gaps:
                if not deeper:
                    stop_reason = "nothing left to try"
                    break
                continue
            log(f"reviser  weak: {', '.join(a.id for a in weak) or 'none'} · deeper: "
                f"{', '.join(a.id for a in deeper) or 'none'}; asking the planner…")
            text = format_revision_input(
                question, [{**self._reviser_view(a), "status": status[a.id]} for a in aspects],
                rnd, self.max_rounds, self.target, total_rel, share)
            rev, ru, rev_cached = await self.reviser.revise(text)
            usage["reviser"].add(ru)
            trace.append({"round": rnd, "role": "reviser", "cached": rev_cached, "input": text,
                          "output": rev.model_dump(), "usage": ru.as_dict()})
            rec["revision"] = {"done": rev.done, "reasoning": rev.reasoning, "cached": rev_cached,
                               "status": status, "updates": [], "new_aspects": [], "ignored": []}
            log(f"reviser  {'(cached) ' if rev_cached else ''}{_short(rev.reasoning, 160)}")
            if rev.done:  # no more rewrites; aspects that are going deeper still continue
                if not deeper:
                    stop_reason = "planner decided done"
                    break
                log("reviser  done with rewrites; deeper aspects continue")
                continue
            by_id = {a.id: a for a in aspects}
            for up in rev.updates:
                a = by_id.get(up.aspect_id)
                # only weak aspects can be rewritten: enforced in code, not just by instruction
                if a is None or status[a.id] != "WEAK":
                    rec["revision"]["ignored"].append(up.aspect_id)
                    continue
                a.bm25_query, a.hyde_abstract, a.needs_search = up.bm25_query, up.hyde_abstract, True
                rec["revision"]["updates"].append(up.model_dump())
                log(f"revise   {a.id}: {_short(up.reason, 90)}")
                log(f"           new bm25: {_short(up.bm25_query, 90)}")
            for na in rev.new_aspects[:1]:
                if len(aspects) >= MAX_ASPECTS:
                    break
                if any(na.name.strip().lower() == a.name.strip().lower()
                       or na.description.strip() == a.description.strip() for a in aspects):
                    log(f"ignored  new aspect {na.name!r}: duplicates an existing aspect")
                    continue
                a = AspectState(f"a{len(aspects) + 1}", na.name, na.description, na.bm25_query,
                                na.hyde_abstract, added_in_round=rnd + 1)
                aspects.append(a)
                rec["revision"]["new_aspects"].append({"id": a.id, **na.model_dump()})
                log(f"add      {a.id} {na.name} | bm25: {_short(na.bm25_query, 80)}")
            for aid in rec["revision"]["ignored"]:
                log(f"ignored  update to {aid} ({status.get(aid, 'unknown aspect')}: only WEAK aspects are rewritten)")
            if not any(a.needs_search or a.go_deeper for a in aspects):
                stop_reason = "no usable revisions"
                break

        total = sum(u.cost_usd for u in usage.values())
        log(f"stop: {stop_reason} · {sum(len(a.verdicts) for a in aspects)} verdicts · ${total:.4f}")
        info = {"plan_cached": plan_cached, "usage": usage, "trace": trace}
        return aspects, rounds, info, stop_reason

    def _reviser_view(self, a: AspectState) -> dict:
        return {"id": a.id, "name": a.name, "description": a.description, "bm25_query": a.bm25_query,
                "hyde_abstract": a.hyde_abstract, "n_relevant": a.n_relevant, "n_checked": len(a.verdicts),
                "good_titles": [self.corpus.get(p)["title"] for p in a.relevant[:N_GOOD_TITLES]],
                "failure_patterns": a.failure_patterns, "missing": a.missing}

    def _final_ranking(self, aspects: list[AspectState], k: int) -> list[AgentHit]:
        # each aspect's lists (one per query version) merge into one aspect list, then across aspects
        aspect_lists = {}
        for a in aspects:
            lists = {f"q{i}": l for i, l in enumerate(a.lists)}
            aspect_lists[a.id] = [h.paper_id for h in rrf(lists, k=self.k)] if len(lists) > 1 else a.lists[0]
        fused = rrf(aspect_lists, k=self.k)

        relevant = {p for a in aspects for p in a.relevant}
        checked = {p for a in aspects for p in a.verdicts}

        def tier(pid):
            if pid in relevant:
                return TIER_RELEVANT
            return TIER_REJECTED if pid in checked else TIER_UNVERIFIED

        # "verified first, then fill": stable sort keeps the RRF order inside each tier
        ordered = sorted(fused, key=lambda h: tier(h.paper_id))[:k]
        return [AgentHit(h.paper_id, h.score, r, h.component_ranks, TIER_NAMES[tier(h.paper_id)])
                for r, h in enumerate(ordered, start=1)]

    def search(self, question: str, k: int = 50) -> tuple[list[AgentHit], dict]:
        t0 = time.perf_counter()
        aspects, rounds, info, stop_reason = self._loop.run_until_complete(self._run(question))
        hits = self._final_ranking(aspects, k)

        total = Usage()
        for u in info["usage"].values():
            total.add(u)
        diag = {
            "mode": "agent" if self.reviser else "verify",
            "planner_model": self.planner.llm.model, "verifier_model": self.verifiers.llm.model,
            "plan_cached": info["plan_cached"], "n_aspects": len(aspects), "n_rounds": len(rounds),
            "stop_reason": stop_reason, "target_relevant": self.target, "max_rounds": self.max_rounds,
            "n_candidates": self.n_candidates,
            "llm_usage": total.as_dict(),
            "llm_usage_by_role": {r: u.as_dict() for r, u in info["usage"].items()},
            "rounds": rounds,
            "aspects": [{"id": a.id, "name": a.name, "description": a.description,
                         "added_in_round": a.added_in_round, "queries": a.query_history,
                         "n_checked": len(a.verdicts), "n_relevant": a.n_relevant,
                         "relevant": a.relevant} for a in aspects],
            # in the final top-k: how many papers each aspect contributed, and by tier
            "final_coverage_by_aspect": {a.name: sum(a.id in h.component_ranks for h in hits) for a in aspects},
            "final_tiers": {t: sum(h.tier == t for h in hits) for t in TIER_NAMES.values()},
            "n_verified_relevant": len({p for a in aspects for p in a.relevant}),
            "latency_s": round(time.perf_counter() - t0, 3), "rrf_k": self.k, "depth": self.depth,
            "trace": info["trace"],  # run_system moves this to trace.jsonl (too big for run_meta)
        }
        return hits, diag
