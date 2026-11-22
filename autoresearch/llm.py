"""Thin wrapper around the OpenAI Responses API: key loading, structured output,
retries, and token/cost accounting for every call."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydantic import BaseModel

from . import PROJECT_ROOT

# USD per 1M tokens: (input, cached input, output). Reasoning tokens are billed as output.
PRICING = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
}

KEY_FILES = ["openai_secrect.txt", "openai_secret.txt"]


def load_api_key() -> str:
    if key := os.getenv("OPENAI_API_KEY"):
        return key.strip()
    for name in KEY_FILES:
        p = PROJECT_ROOT / name
        if p.exists():
            return p.read_text().strip()
    raise RuntimeError("No OpenAI key: set OPENAI_API_KEY or put it in openai_secrect.txt")


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        for f in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "calls"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["cost_usd"] = round(self.cost_usd, 6)
        return d


def usage_from_response(resp, model: str) -> Usage:
    u = resp.usage
    cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
    reasoning = getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0
    p_in, p_cached, p_out = PRICING.get(model, (0.0, 0.0, 0.0))
    cost = ((u.input_tokens - cached) * p_in + cached * p_cached + u.output_tokens * p_out) / 1e6
    return Usage(u.input_tokens, cached, u.output_tokens, reasoning, 1, cost)


@dataclass
class LLM:
    model: str = "gpt-5.6-luna"
    reasoning_effort: str | None = "low"
    total: Usage = field(default_factory=Usage)

    def __post_init__(self):
        from openai import AsyncOpenAI

        if self.model not in PRICING:
            print(f"warning: no pricing for {self.model}; cost will show as 0")
        self.client = AsyncOpenAI(api_key=load_api_key(), max_retries=5)

    async def parse(self, *, instructions: str, input: str, schema: type[BaseModel],
                    cache_key: str | None = None) -> tuple[BaseModel, Usage]:
        """One structured-output call. Returns the parsed object and this call's usage."""
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if cache_key:  # route requests sharing a prefix to the same cache
            kwargs["prompt_cache_key"] = cache_key
        resp = await self.client.responses.parse(
            model=self.model, instructions=instructions, input=input,
            text_format=schema, store=False, **kwargs,
        )
        usage = usage_from_response(resp, self.model)
        self.total.add(usage)
        if resp.output_parsed is None:
            raise ValueError(f"Model returned no parseable output: {resp.output_text[:200]!r}")
        return resp.output_parsed, usage
