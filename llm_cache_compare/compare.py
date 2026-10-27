"""Replay the same agent workloads against several providers and log every request.

  python -m llm_cache_compare.compare --dry-run      # checks each key + caching; prints cost estimate
  python -m llm_cache_compare.compare                # the full comparison (all enabled providers)
  python -m llm_cache_compare.compare --quick        # tiny version of everything (pipeline check)
  python -m llm_cache_compare.compare --only groq,openai

Keys come from the environment or .env. Output: results/compare/<run_id>/<provider>.jsonl,
then `python -m llm_cache_compare.report` builds the summary used by the dashboard.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llm_cache_compare.adapters import CallResult, ProviderClient
from llm_cache_compare.budget import BudgetExceeded, BudgetGuard
from llm_cache_compare.providers import Provider, cost_from_usage, load_dotenv, load_providers
from llm_cache_compare.workload import ToolOutputGenerator

MAX_CONCURRENCY = 8
MAX_TOTAL_BUDGET_USD = 10.0
PAUSE_SCALE = 1.0  # tests set 0 to skip the polite pauses between requests
LOOP_ASK = "Reply with one short sentence: the next action you will take."
SYSTEM = ("Run {nonce}. You are a coding agent working in a large repository. You read tool "
          "output and decide the next step. Be terse.")


async def _pause(seconds: float) -> None:
    if PAUSE_SCALE > 0:
        await asyncio.sleep(seconds * PAUSE_SCALE)


def tool_tokens_for_turn(turn: int, lo: int, hi: int) -> int:
    return lo if lo == hi else lo + (turn * 7919) % (hi - lo + 1)


def load_config(path: Path | str, quick: bool = False) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if quick:
        for section, over in (cfg.get("quick") or {}).items():
            cfg[section] = {**cfg[section], **over}
    if max(cfg["concurrency"]["levels"]) > MAX_CONCURRENCY:
        raise ValueError(f"concurrency levels must be <= {MAX_CONCURRENCY}")
    if cfg["budget"]["total_usd"] > MAX_TOTAL_BUDGET_USD:
        raise ValueError(f"budget.total_usd must be <= {MAX_TOTAL_BUDGET_USD}")
    return cfg


class TwoLevelBudget:
    """Every request must fit both the provider's cap and the shared lifetime cap."""

    def __init__(self, provider_guard: BudgetGuard, shared: BudgetGuard) -> None:
        self.provider, self.shared = provider_guard, shared

    async def reserve(self, estimate: float) -> float:
        await self.provider.reserve(estimate)
        try:
            await self.shared.reserve(estimate)
        except BudgetExceeded:
            await self.provider.settle(estimate, 0.0)
            raise
        return estimate

    async def settle(self, reserved: float, actual: float) -> None:
        await self.provider.settle(reserved, actual)
        await self.shared.settle(reserved, actual)


class Log:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path, self.run_id = path, run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self.count = self.errors = 0

    def write(self, r: CallResult) -> None:
        rec = {"run_id": self.run_id, "logged_at": time.time(), **r.to_dict()}
        rec["output_text"] = rec["output_text"][:300]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.count += 1
        self.errors += r.error is not None


class ProviderAborted(RuntimeError):
    pass


class ProviderRun:
    """All tests for one provider, in a fixed interleaved order."""

    def __init__(self, provider: Provider, client: ProviderClient, cfg: dict[str, Any],
                 log: Log, gen: ToolOutputGenerator | None = None) -> None:
        self.p, self.client, self.cfg, self.log = provider, client, cfg, log
        self.gen = gen or ToolOutputGenerator()
        self.consecutive_errors = 0
        self.spent = 0.0

    def _say(self, msg: str) -> None:
        print(f"[{self.p.name:<9}] {msg}", flush=True)

    async def _call(self, system: str, messages: list[dict[str, Any]], *, max_tokens: int,
                    test: str, tags: dict[str, Any], cache_key: str | None = None
                    ) -> CallResult:
        r = await self.client.call(system, messages, max_tokens=max_tokens, test=test,
                                   tags=tags, cache_key=cache_key)
        self.log.write(r)
        self.spent += r.cost_usd or 0.0
        if r.error:
            self.consecutive_errors += 1
            self._say(f"error in {test} {tags}: {r.error[:160]}")
            if r.status_code in (400, 401, 402, 403, 404, 413) or self.consecutive_errors >= 5:
                raise ProviderAborted(r.error)
        else:
            self.consecutive_errors = 0
        return r

    async def loop(self, rep: int, bug: bool) -> None:
        c = self.cfg["loop"]
        lo, hi = c["tool_tokens"]
        test = "loop_bug" if bug else "loop_healthy"
        nonce = uuid.uuid4().hex
        base = SYSTEM.format(nonce=nonce)
        messages: list[dict[str, str]] = []
        cost = 0.0
        for turn in range(1, c["turns"] + 1):
            tool = self.gen.make(tool_tokens_for_turn(turn, lo, hi), turn)
            messages.append({"role": "user", "content": f"Tool output:\n{tool}\n\n{LOOP_ASK}"})
            system = base
            if bug:  # the bug: a fresh timestamp in the system prompt on every request
                system = f"{base}\nCurrent time: {datetime.now(UTC).isoformat()}"
            r = await self._call(system, messages, max_tokens=c["max_tokens"], test=test,
                                 tags={"rep": rep, "turn": turn}, cache_key=nonce)
            cost += r.cost_usd or 0.0
            messages.append({"role": "assistant",
                             "content": (r.output_text or "").strip() or "Continuing."})
        self._say(f"{test:<12} rep {rep}: {c['turns']} turns, ${cost:.4f}")

    async def long_context(self, rep: int) -> None:
        c = self.cfg["long_context"]
        for size in c["sizes"]:
            nonce = uuid.uuid4().hex
            system = SYSTEM.format(nonce=nonce)
            doc = self.gen.make(size, 1000 * rep + size % 997)
            questions = ["How many lines in the log start with 'PASS'? Answer with a number.",
                         "Name one file that appears in the log. Answer with the path only.",
                         "Which HTTP status code appears in the log? Answer with the number."]
            for i in range(1 + c["warm_repeats"]):
                q = questions[i % len(questions)]
                # document and question as separate parts: the document is the shared prefix
                await self._call(system, [{"role": "user", "content": [doc, q]}],
                                 max_tokens=c["max_tokens"], test="long_context",
                                 tags={"rep": rep, "size": size,
                                       "phase": "cold" if i == 0 else "warm", "attempt": i},
                                 cache_key=nonce)
                await _pause(1.0)
        self._say(f"long_context rep {rep}: sizes {c['sizes']}")

    async def concurrency(self, rep: int) -> None:
        c = self.cfg["concurrency"]
        for level in c["levels"]:
            async def one(i: int, level: int = level) -> CallResult:
                system = SYSTEM.format(nonce=uuid.uuid4().hex)
                context = self.gen.make(c["prompt_tokens"], 500 + i)
                ask = ("Using the log above as context, write about 400 words explaining how a "
                       "team should triage failing CI builds. Plain prose, no lists.")
                msgs = [{"role": "user", "content": f"{context}\n\n{ask}"}]
                return await self._call(system, msgs, max_tokens=c["max_tokens"],
                                        test="concurrency",
                                        tags={"rep": rep, "level": level, "stream": i})
            t0 = time.perf_counter()
            results = await asyncio.gather(*(one(i) for i in range(level)))
            wall = time.perf_counter() - t0
            out = sum(r.completion_tokens or 0 for r in results)
            self._say(f"concurrency  rep {rep}: {level} streams, {out / wall:,.0f} tok/s total")
            await _pause(2.0)

    async def run_all(self) -> str:
        reps = max(self.cfg["loop"]["reps"], self.cfg["long_context"]["reps"],
                   self.cfg["concurrency"]["reps"])
        try:
            for rep in range(1, reps + 1):
                if rep <= self.cfg["loop"]["reps"]:
                    await self.loop(rep, bug=False)
                    await self.loop(rep, bug=True)
                if rep <= self.cfg["long_context"]["reps"]:
                    await self.long_context(rep)
                if rep <= self.cfg["concurrency"]["reps"]:
                    await self.concurrency(rep)
        except ProviderAborted as e:
            self._say(f"STOPPED: {str(e)[:200]}")
            return "aborted"
        except BudgetExceeded as e:
            self._say(f"STOPPED (budget): {e}")
            return "budget"
        self._say(f"done: {self.log.count} requests, {self.log.errors} errors, "
                  f"${self.spent:.4f}")
        return "ok"


# ---------------- cost estimate (no API calls) ----------------

def estimate_cost(p: Provider, cfg: dict[str, Any], safety: float = 1.4) -> dict[str, float]:
    """Model the token flow of the full run assuming caching works perfectly in the healthy
    loop and not at all in the buggy one. `safety` covers tokenizer differences."""
    writes = p.prices.cache_write > p.prices.input  # writes billed above input (Anthropic, OpenAI)

    def cost(prompt: float, cached: float, new: float, out: float) -> float:
        return cost_from_usage(p, prompt_tokens=int(prompt), cached_tokens=int(cached),
                               cache_write_tokens=int(new) if writes else 0,
                               completion_tokens=int(out))

    c = cfg["loop"]
    lo, hi = c["tool_tokens"]
    sys_t, reply_t, out_t = 60, 40, 60 if p.api != "openai_chat" else 250
    healthy = bug = 0.0
    prev = 0.0
    prompt = float(sys_t)
    for turn in range(1, c["turns"] + 1):
        prompt += (tool_tokens_for_turn(turn, lo, hi) + 25) * safety
        healthy += cost(prompt, prev, prompt - prev, out_t)
        bug += cost(prompt, 0, prompt, out_t)
        prompt += reply_t
        prev = prompt
    lc = cfg["long_context"]
    long_ctx = sum(cost(s * safety, 0, s * safety, 30)
                   + lc["warm_repeats"] * cost(s * safety, s * safety, 0, 30)
                   for s in lc["sizes"])
    cc = cfg["concurrency"]
    conc = sum(lvl * cost(cc["prompt_tokens"] * safety, 0, 0, cc["max_tokens"] * 0.6)
               for lvl in cc["levels"])
    out = {"loop_healthy": healthy * c["reps"], "loop_bug": bug * c["reps"],
           "long_context": long_ctx * lc["reps"], "concurrency": conc * cc["reps"]}
    out["total"] = sum(out.values())
    return out


def print_estimates(providers: list[Provider], cfg: dict[str, Any]) -> float:
    print("\nEstimated cost at list prices (assumes ~40% more tokens than our calibration):")
    print(f"{'provider':<10} {'healthy':>9} {'bug':>9} {'long ctx':>9} {'concur.':>9} "
          f"{'total':>9}")
    grand = 0.0
    for p in providers:
        e = estimate_cost(p, cfg)
        grand += e["total"]
        print(f"{p.name:<10} {e['loop_healthy']:>9.3f} {e['loop_bug']:>9.3f} "
              f"{e['long_context']:>9.3f} {e['concurrency']:>9.3f} {e['total']:>9.3f}")
    print(f"{'all':<10} {'':>39} {grand:>9.3f}\n")
    return grand


# ---------------- dry run: 2 tiny requests per provider ----------------

async def check_provider(p: Provider, budget: TwoLevelBudget) -> bool:
    """Send the same ~5K-token document twice: proves the key works and shows caching
    (5K clears every provider's minimum cacheable size, including Gemini's 4,096)."""
    try:
        client = ProviderClient(p, budget, max_attempts=4)
    except RuntimeError as e:
        print(f"[{p.name:<9}] MISSING KEY: {e}")
        return False
    try:
        nonce = uuid.uuid4().hex
        system = SYSTEM.format(nonce=nonce)
        doc = ToolOutputGenerator().make(5000, 7)
        rows = []
        for q in ("Reply with the word OK.", "Reply with the word YES."):
            rows.append(await client.call(system, [{"role": "user", "content": [doc, q]}],
                                          max_tokens=300, test="check", tags={},
                                          cache_key=nonce))
            await _pause(5.0)
        a, b = rows
        if a.error or b.error:
            print(f"[{p.name:<9}] FAILED: {(a.error or b.error)[:300]}")
            return False
        share = (b.cached_tokens or 0) / b.prompt_tokens if b.prompt_tokens else 0
        print(f"[{p.name:<9}] OK  {p.model}: prompt {b.prompt_tokens} tok, "
              f"cached on 2nd call {b.cached_tokens} ({share:.0%}), ttft {a.ttft_s or 0:.2f}s -> "
              f"{b.ttft_s or 0:.2f}s, reply {b.output_text.strip()[:30]!r}, "
              f"cost ${(a.cost_usd or 0) + (b.cost_usd or 0):.5f}")
        print(f"[{p.name:<9}]     usage reported on 2nd call: {json.dumps(b.raw_usage)[:400]}")
        if not b.cached_tokens:
            print(f"[{p.name:<9}]     note: no cache hit on the repeat ({p.caching}). "
                  f"Some providers need a few seconds or a longer prompt; the full run will show.")
        return True
    finally:
        await client.aclose()


# ---------------- main ----------------

def _select(providers: list[Provider], only: str | None) -> list[Provider]:
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        unknown = wanted - {p.name for p in providers}
        if unknown:
            sys.exit(f"unknown provider(s): {', '.join(sorted(unknown))}")
        return [p for p in providers if p.name in wanted]
    return [p for p in providers if p.enabled]


async def main_async(args: argparse.Namespace) -> int:
    set_names = load_dotenv(args.env)
    if set_names:
        print(f"loaded from {args.env}: {', '.join(set_names)}")
    cfg = load_config(args.config, quick=args.quick)
    providers = _select(load_providers(args.providers), args.only)
    if not providers:
        sys.exit("no providers selected")
    out_root = Path(args.out)
    shared = BudgetGuard(run_cap_usd=cfg["budget"]["total_usd"],
                         total_cap_usd=cfg["budget"]["total_usd"],
                         ledger_path=out_root / "spend_ledger.json")
    print(f"budget: ${shared.total_spent:.4f} spent so far of ${cfg['budget']['total_usd']:.2f} "
          f"lifetime; ${cfg['budget']['per_provider_usd']:.2f} max per provider per run")

    if args.dry_run:
        grand = print_estimates(providers, cfg)
        ok = []
        for p in providers:
            guard = BudgetGuard(run_cap_usd=0.05, total_cap_usd=1e9,
                                ledger_path=out_root / ".dry_run_ledger.json")
            ok.append(await check_provider(p, TwoLevelBudget(guard, shared)))
        remaining = cfg["budget"]["total_usd"] - shared.total_spent
        print(f"\n{sum(ok)}/{len(ok)} providers ready. Estimated full run ${grand:.2f}; "
              f"budget left ${remaining:.2f}.")
        return 0 if all(ok) else 1

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{'quick' if args.quick else 'compare'}-" \
             f"{uuid.uuid4().hex[:6]}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "started_at": time.time(), "quick": args.quick, "config": cfg,
            "providers": [{k: v for k, v in vars(p).items() if k != "prices"}
                          | {"prices": vars(p.prices)} for p in providers]}
    for m in meta["providers"]:
        if m.get("long_context_prices") is not None:
            m["long_context_prices"] = vars(m["long_context_prices"])
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    print_estimates(providers, cfg)
    print(f"run {run_id} -> {run_dir}\n")

    runs, clients = [], []
    for p in providers:
        try:
            client = ProviderClient(p, TwoLevelBudget(
                BudgetGuard(run_cap_usd=cfg["budget"]["per_provider_usd"], total_cap_usd=1e9,
                            ledger_path=run_dir / f".{p.name}_ledger.json"), shared))
        except RuntimeError as e:
            print(f"[{p.name:<9}] skipped: {e}")
            continue
        clients.append(client)
        runs.append(ProviderRun(p, client, cfg, Log(run_dir / f"{p.name}.jsonl", run_id)))
    t0 = time.time()
    try:
        statuses = await asyncio.gather(*(r.run_all() for r in runs))
    finally:
        for c in clients:
            await c.aclose()
    meta.update(finished_at=time.time(),
                status={r.p.name: s for r, s in zip(runs, statuses, strict=True)},
                spent={r.p.name: round(r.spent, 6) for r in runs})
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    total = sum(r.spent for r in runs)
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min. spent ${total:.4f} this run "
          f"(lifetime ${shared.total_spent:.4f}).")
    print(f"next: python -m llm_cache_compare.report {run_dir}")
    return 0 if all(s == "ok" for s in statuses) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="check keys + caching with 2 tiny requests each; print cost estimate")
    ap.add_argument("--quick", action="store_true", help="tiny version of every test")
    ap.add_argument("--only", help="comma-separated provider names, e.g. groq,openai")
    ap.add_argument("--config", default="config/compare.yaml")
    ap.add_argument("--providers", default="providers.yaml")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--out", default="results/compare")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
