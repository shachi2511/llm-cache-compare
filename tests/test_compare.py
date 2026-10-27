"""Cross-provider comparison: adapters for all four API formats, cost model, full quick run.
Everything runs against tests/mock_providers.py -- no network, no keys, no spend."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest
import yaml

from llm_cache_compare.adapters import ProviderClient, decode_tps
from llm_cache_compare.budget import BudgetGuard
from llm_cache_compare.compare import estimate_cost, load_config, main_async
from llm_cache_compare.providers import (
    Prices,
    Provider,
    cost_from_usage,
    load_dotenv,
    load_providers,
)
from tests.mock_providers import serve

ROOT = Path(__file__).resolve().parents[1]
APIS = {"openai_chat": "/openai/v1", "openai_responses": "/v1", "anthropic": "",
        "gemini": "/v1beta"}


@pytest.fixture(autouse=True)
def no_pauses(monkeypatch):
    monkeypatch.setattr("llm_cache_compare.compare.PAUSE_SCALE", 0.0)


@pytest.fixture()
def mock():
    server, state = serve(0)
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()


def provider(api: str, base: str, **kw) -> Provider:
    return Provider(name=kw.pop("name", api), label=api, api=api, base_url=base + APIS[api],
                    key_env="AGENTLOAD_TEST_KEY", model="m",
                    prices=Prices(input=1.0, cached_read=0.1, cache_write=1.25, output=5.0),
                    options={"cache_control": {"type": "ephemeral"}} if api == "anthropic"
                    else {}, **kw)


def budget(tmp_path: Path, cap: float = 1.0) -> BudgetGuard:
    return BudgetGuard(run_cap_usd=cap, total_cap_usd=cap, ledger_path=tmp_path / "l.json")


# ---------- config + cost model ----------

def test_real_provider_file_loads():
    providers = load_providers(ROOT / "providers.yaml")
    names = {p.name for p in providers if p.enabled}
    assert {"openai", "anthropic", "gemini"} <= names
    for p in providers:
        assert p.prices.cached_read <= p.prices.input
        assert p.price_source.startswith("https://")


def test_cost_from_usage_and_long_context_tier():
    p = Provider(name="a", label="A", api="anthropic", base_url="x", key_env="K", model="m",
                 prices=Prices(0.10, 0.01, 0.125, 0.50), long_context_threshold=100_000,
                 long_context_prices=Prices(0.50, 0.05, 0.625, 2.50))
    c = cost_from_usage(p, prompt_tokens=10_000, cached_tokens=6_000, cache_write_tokens=3_000,
                        completion_tokens=100)
    assert c == pytest.approx((1_000 * 0.10 + 6_000 * 0.01 + 3_000 * 0.125 + 100 * 0.5) / 1e6)
    big = cost_from_usage(p, prompt_tokens=120_000, completion_tokens=0)
    assert big == pytest.approx(120_000 * 0.50 / 1e6)


def test_dotenv_never_overrides_and_never_returns_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# keys\nNEW_KEY='abc'\nexport OTHER=xyz\nEXISTING=new\n")
    monkeypatch.setenv("EXISTING", "old")
    monkeypatch.delenv("NEW_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    assert load_dotenv(env) == ["NEW_KEY", "OTHER"]
    import os
    assert os.environ["NEW_KEY"] == "abc" and os.environ["EXISTING"] == "old"


def test_estimate_bug_costs_more_than_healthy_for_every_provider():
    cfg = load_config(ROOT / "config/compare.yaml")
    for p in load_providers(ROOT / "providers.yaml"):
        e = estimate_cost(p, cfg)
        assert e["loop_bug"] > e["loop_healthy"], p.name
        if p.prices.cached_read <= 0.1 * p.prices.input:  # 90% cache discount -> big gap
            assert e["loop_bug"] > 3 * e["loop_healthy"], p.name
    total = sum(estimate_cost(p, cfg)["total"]
                for p in load_providers(ROOT / "providers.yaml") if p.enabled)
    assert total < cfg["budget"]["total_usd"]


def test_decode_tps_ignores_gemini_hidden_thinking():
    row = {"api": "gemini", "completion_tokens": 150, "reasoning_tokens": 50, "ttft_s": 1.0,
           "last_token_s": 2.0, "text_chunks": 10}
    assert decode_tps(row) == pytest.approx(100.0)
    assert decode_tps({**row, "text_chunks": 2}) is None


# ---------- adapters against the fake APIs ----------

@pytest.mark.parametrize("api", sorted(APIS))
def test_adapter_streams_usage_and_cache(api, mock, tmp_path, monkeypatch):
    base, state = mock
    monkeypatch.setenv("AGENTLOAD_TEST_KEY", "k")
    p = provider(api, base)
    doc = "line of tool output number 12345\n" * 200

    async def go():
        client = ProviderClient(p, budget(tmp_path))
        try:
            a = await client.call("sys", [{"role": "user", "content": doc + "q1"}],
                                  max_tokens=50, test="t")
            b = await client.call("sys", [{"role": "user", "content": doc + "q2"}],
                                  max_tokens=50, test="t")
        finally:
            await client.aclose()
        return a, b

    a, b = asyncio.run(go())
    for r in (a, b):
        assert r.error is None, r.error
        assert r.output_text.startswith("Next I will")
        assert r.ttft_s is not None and r.total_s >= r.ttft_s
        assert r.prompt_tokens > 2000 and r.cost_usd > 0
    assert (a.cached_tokens or 0) == 0
    assert b.cached_tokens / b.prompt_tokens > 0.9      # warm repeat is served from cache
    assert b.cost_usd < a.cost_usd
    if api == "anthropic":
        assert a.cache_write_tokens == a.prompt_tokens  # Anthropic bills the first write
        assert state.bodies[0]["cache_control"] == {"type": "ephemeral"}


def test_rate_limit_is_retried_and_counted(tmp_path, monkeypatch):
    server, _state = serve(0, fail_first=2)
    monkeypatch.setenv("AGENTLOAD_TEST_KEY", "k")
    p = provider("openai_chat", f"http://127.0.0.1:{server.server_port}")

    async def go():
        client = ProviderClient(p, budget(tmp_path), base_backoff_s=0.01)
        try:
            return await client.call("s", [{"role": "user", "content": "hi"}], max_tokens=5,
                                      test="t")
        finally:
            await client.aclose()

    r = asyncio.run(go())
    server.shutdown()
    assert r.error is None and r.attempts == 3 and r.rate_limited == 2


def test_budget_hold_blocks_overspend(tmp_path, monkeypatch, mock):
    from llm_cache_compare.budget import BudgetExceeded
    monkeypatch.setenv("AGENTLOAD_TEST_KEY", "k")
    p = provider("openai_chat", mock[0])

    async def go():
        client = ProviderClient(p, budget(tmp_path, cap=1e-7))
        try:
            await client.call("s", [{"role": "user", "content": "x" * 5000}], max_tokens=500,
                              test="t")
        finally:
            await client.aclose()

    with pytest.raises(BudgetExceeded):
        asyncio.run(go())
    assert mock[1].requests == 0  # refused before anything was sent


# ---------- the whole comparison, quick mode, all four APIs at once ----------

def test_quick_compare_end_to_end(mock, tmp_path, monkeypatch):
    base, _ = mock
    monkeypatch.setenv("AGENTLOAD_TEST_KEY", "k")
    providers = {"providers": [
        {"name": api.replace("_", "-"), "api": api, "base_url": base + APIS[api],
         "key_env": "AGENTLOAD_TEST_KEY", "model": "m",
         "options": {"cache_control": {"type": "ephemeral"}} if api == "anthropic" else {},
         "prices": {"input": 1.0, "cached_read": 0.1, "cache_write": 1.25, "output": 5.0},
         "price_source": "https://example.com"} for api in APIS]}
    (tmp_path / "providers.yaml").write_text(yaml.safe_dump(providers))
    out = tmp_path / "out"
    args = argparse.Namespace(dry_run=False, quick=True, only=None,
                              config=str(ROOT / "config/compare.yaml"),
                              providers=str(tmp_path / "providers.yaml"),
                              env=str(tmp_path / "missing.env"), out=str(out))
    assert asyncio.run(main_async(args)) == 0

    run_dir = next(d for d in out.iterdir() if d.is_dir())
    meta = json.loads((run_dir / "run.json").read_text())
    assert set(meta["status"].values()) == {"ok"}
    assert "AGENTLOAD_TEST_KEY" not in json.dumps(meta["config"])
    for api in APIS:
        rows = [json.loads(x) for x in (run_dir / f"{api.replace('_', '-')}.jsonl").open()]
        assert all(r["error"] is None for r in rows)
        tests = {r["test"] for r in rows}
        assert tests == {"loop_healthy", "loop_bug", "long_context", "concurrency"}
        healthy = [r for r in rows if r["test"] == "loop_healthy"]
        bug = [r for r in rows if r["test"] == "loop_bug"]
        assert len(healthy) == len(bug) == 4
        # healthy loop reuses the cache from turn 2; the buggy loop never does
        assert all(r["cached_tokens"] > 0 for r in healthy[1:])
        assert all(r["cached_tokens"] == 0 for r in bug)
        assert sum(r["cost_usd"] for r in bug) > sum(r["cost_usd"] for r in healthy)

    # the summary that feeds the dashboard
    from llm_cache_compare.report import summarize
    summary = summarize([run_dir], include_quick=True)
    assert summary["totals"]["providers"] == 4 and summary["totals"]["errors"] == 0
    for p in summary["providers"]:
        assert p["loops"]["bug_multiplier"]["mean"] > 1
        assert p["loops"]["loop_healthy"]["reuse_median"] > 0.8
        assert p["long_context"][0]["warm_cached_share"] > 0.8
        assert [c["level"] for c in p["concurrency"]] == [1, 2]
    json.dumps(summary)

    # dry run against the same fakes: keys work, cache hit on the repeat
    args.dry_run = True
    assert asyncio.run(main_async(args)) == 0


def test_document_and_question_are_separate_cacheable_parts(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTLOAD_TEST_KEY", "k")
    msgs = [{"role": "user", "content": ["DOC", "Q"]}]
    ant = ProviderClient(provider("anthropic", "http://x"), budget(tmp_path))
    _, body = ant._request("sys", msgs, 10)
    blocks = body["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "DOC", "cache_control": {"type": "ephemeral"}}
    assert blocks[1] == {"type": "text", "text": "Q"}
    oai = ProviderClient(provider("openai_responses", "http://x"), budget(tmp_path))
    _, body = oai._request("sys", msgs, 10, cache_key="conv1")
    assert [p["text"] for p in body["input"][0]["content"]] == ["DOC", "Q"]
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in body["input"][0]["content"][1]
    assert body["prompt_cache_key"] == "conv1"
    gem = ProviderClient(provider("gemini", "http://x"), budget(tmp_path))
    _, body = gem._request("sys", msgs, 10)
    assert body["contents"][0]["parts"] == [{"text": "DOC"}, {"text": "Q"}]
    chat = ProviderClient(provider("openai_chat", "http://x"), budget(tmp_path))
    _, body = chat._request("sys", msgs, 10)
    assert body["messages"][1]["content"] == "DOC\n\nQ"
    for c in (ant, oai, gem, chat):
        asyncio.run(c.aclose())
