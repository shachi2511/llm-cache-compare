"""Provider registry for cross-provider comparisons.

A provider = one API + one model + its list prices. Costs are computed from the token counts
each API reports (most providers don't return a charge), using `cost_from_usage`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

APIS = {"openai_chat", "openai_responses", "anthropic", "gemini"}


@dataclass(frozen=True)
class Prices:
    input: float          # uncached input, per 1M tokens
    cached_read: float    # input served from cache
    cache_write: float    # input written to cache (Anthropic bills writes above input)
    output: float         # output, including reasoning/thinking tokens

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prices:
        return cls(**{k: float(d[k]) for k in ("input", "cached_read", "cache_write", "output")})

    @property
    def dearest_input(self) -> float:
        return max(self.input, self.cache_write)


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    api: str
    base_url: str
    key_env: str
    model: str
    prices: Prices
    enabled: bool = True
    caching: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    long_context_threshold: int | None = None
    long_context_prices: Prices | None = None
    price_source: str = ""
    min_interval_s: float = 0.0   # pacing for strict free tiers: min gap between requests

    def prices_for(self, prompt_tokens: int) -> Prices:
        if (self.long_context_threshold and self.long_context_prices
                and prompt_tokens > self.long_context_threshold):
            return self.long_context_prices
        return self.prices

    @property
    def worst_prices(self) -> Prices:
        return self.long_context_prices or self.prices

    def api_key(self) -> str | None:
        return os.environ.get(self.key_env) or None


def load_providers(path: Path | str = "providers.yaml") -> list[Provider]:
    raw = yaml.safe_load(Path(path).read_text())
    out: list[Provider] = []
    for p in raw["providers"]:
        if p["api"] not in APIS:
            raise ValueError(f"{p['name']}: unknown api {p['api']!r}; known: {sorted(APIS)}")
        lc = p.get("long_context") or {}
        out.append(Provider(
            name=p["name"], label=p.get("label", p["name"]), api=p["api"],
            base_url=p["base_url"].rstrip("/"), key_env=p["key_env"], model=p["model"],
            prices=Prices.from_dict(p["prices"]), enabled=bool(p.get("enabled", True)),
            caching=p.get("caching", ""), options=dict(p.get("options") or {}),
            long_context_threshold=lc.get("threshold_tokens"),
            long_context_prices=Prices.from_dict(lc["prices"]) if lc.get("prices") else None,
            price_source=p.get("price_source", ""),
            min_interval_s=float(p.get("min_interval_s", 0.0)),
        ))
    names = [p.name for p in out]
    if len(names) != len(set(names)):
        raise ValueError("provider names must be unique")
    return out


def load_dotenv(path: Path | str = ".env") -> list[str]:
    """Minimal .env reader (KEY=value lines). Never overrides variables already set.
    Returns the names it set -- never the values."""
    p = Path(path)
    if not p.exists():
        return []
    set_names = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            set_names.append(key)
    return set_names


def cost_from_usage(provider: Provider, *, prompt_tokens: int, cached_tokens: int = 0,
                    cache_write_tokens: int = 0, completion_tokens: int = 0) -> float:
    """USD for one request. prompt_tokens = ALL input tokens (cached + written + plain)."""
    price = provider.prices_for(prompt_tokens)
    plain = max(prompt_tokens - cached_tokens - cache_write_tokens, 0)
    return (plain * price.input + cached_tokens * price.cached_read
            + cache_write_tokens * price.cache_write + completion_tokens * price.output) / 1e6
