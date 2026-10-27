"""Turn compare-run logs into one summary.json (feeds the comparison dashboard) + a text table.

  python -m llm_cache_compare.report                       # every run under results/compare/
  python -m llm_cache_compare.report results/compare/<run> # specific run(s)

Every number in the summary is computed from the JSONL logs; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_cache_compare.adapters import decode_tps, streamed_tokens

Row = dict[str, Any]


def _med(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(x, nd)


def load_runs(paths: list[Path], include_quick: bool = False
              ) -> tuple[dict[str, list[Row]], dict[str, dict[str, Any]], list]:
    rows: dict[str, list[Row]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    runs = []
    for run_dir in paths:
        info = json.loads((run_dir / "run.json").read_text())
        if info.get("quick") and not include_quick:
            print(f"skipping quick run {run_dir.name}")
            continue
        runs.append({k: info.get(k) for k in ("run_id", "started_at", "finished_at", "status",
                                              "spent")})
        for p in info["providers"]:
            meta[p["name"]] = p
        for f in sorted(run_dir.glob("*.jsonl")):
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    row["_run"] = info["run_id"]
                    rows[row["provider"]].append(row)
    return rows, meta, runs


def summarize_loops(rows: list[Row]) -> dict[str, Any]:
    by: dict[tuple[str, str, int], list[Row]] = defaultdict(list)
    for r in rows:
        if r["test"] in ("loop_healthy", "loop_bug"):
            by[(r["_run"], r["test"], r["tags"]["rep"])].append(r)
    out: dict[str, Any] = {}
    sessions: dict[str, dict[tuple[str, int], float]] = {"loop_healthy": {}, "loop_bug": {}}
    for test in ("loop_healthy", "loop_bug"):
        reps = [sorted(v, key=lambda r: r["tags"]["turn"])
                for (run, t, rep), v in by.items() if t == test]
        if not reps:
            continue
        turns = max(len(x) for x in reps)
        per_turn = []
        for i in range(turns):
            cells = [rep[i] for rep in reps if i < len(rep) and rep[i]["error"] is None]
            reuse = []
            for rep in reps:
                if 0 < i < len(rep) and rep[i]["error"] is None and rep[i - 1]["prompt_tokens"]:
                    reuse.append((rep[i]["cached_tokens"] or 0) / rep[i - 1]["prompt_tokens"])
            per_turn.append({
                "turn": i + 1,
                "cost": _r(_mean([c["cost_usd"] for c in cells]), 8),
                "prompt_tokens": _r(_mean([c["prompt_tokens"] for c in cells]), 0),
                "cached_share": _r(_mean([(c["cached_tokens"] or 0) / c["prompt_tokens"]
                                          for c in cells if c["prompt_tokens"]]), 4),
                "reuse": _r(_mean(reuse), 4) if i else None,
                "ttft": _r(_med([c["ttft_s"] for c in cells]), 3),
            })
        for (run, t, rep), v in by.items():
            # only complete sessions count: every turn present, none failed
            if t == test and len(v) == turns and all(r["error"] is None for r in v):
                sessions[test][(run, rep)] = sum(r["cost_usd"] for r in v)
        totals = list(sessions[test].values())
        ok = [r for rep in reps for r in rep if r["error"] is None]
        out[test] = {
            "reps": len(reps), "complete_sessions": len(totals), "turns": turns,
            "per_turn": per_turn,
            "session_cost_mean": _r(_mean(totals)),
            "session_cost_min": _r(min(totals, default=None)),
            "session_cost_max": _r(max(totals, default=None)),
            "reuse_median": _r(_med([p["reuse"] for p in per_turn[1:]]), 4),
            "ttft_median": _r(_med([r["ttft_s"] for r in ok]), 3),
            "ttft_p95": _r(_pct([r["ttft_s"] for r in ok], 95), 3),
            "first_turn_cost": per_turn[0]["cost"], "last_turn_cost": per_turn[-1]["cost"],
            "first_prompt_tokens": per_turn[0]["prompt_tokens"],
            "last_prompt_tokens": per_turn[-1]["prompt_tokens"],
        }
    mult = [sessions["loop_bug"][k] / sessions["loop_healthy"][k]
            for k in sessions["loop_bug"] if k in sessions["loop_healthy"]
            and sessions["loop_healthy"][k] > 0]
    out["bug_multiplier"] = {"mean": _r(_mean(mult), 3), "min": _r(min(mult, default=None), 3),
                             "max": _r(max(mult, default=None), 3), "n": len(mult)}
    return out


def _pct(xs: list[float | None], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_long_context(rows: list[Row]) -> list[dict[str, Any]]:
    by: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for r in rows:
        if r["test"] == "long_context" and r["error"] is None:
            by[(r["tags"]["size"], r["tags"]["phase"])].append(r)
    out = []
    for size in sorted({s for s, _ in by}):
        cold, warm = by.get((size, "cold"), []), by.get((size, "warm"), [])
        c, w = _med([r["ttft_s"] for r in cold]), _med([r["ttft_s"] for r in warm])
        out.append({
            "size": size, "n_cold": len(cold), "n_warm": len(warm),
            "prompt_tokens": _r(_med([r["prompt_tokens"] for r in cold + warm]), 0),
            "ttft_cold": _r(c, 3), "ttft_warm": _r(w, 3),
            "speedup": _r(c / w, 2) if c and w else None,
            "warm_cached_share": _r(_med([(r["cached_tokens"] or 0) / r["prompt_tokens"]
                                          for r in warm if r["prompt_tokens"]]), 4),
            "warm_hit_rate": _r(sum(1 for r in warm if (r["cached_tokens"] or 0) > 0)
                                / len(warm), 3) if warm else None,
            "cost_cold": _r(_mean([r["cost_usd"] for r in cold]), 8),
            "cost_warm": _r(_mean([r["cost_usd"] for r in warm]), 8),
        })
    return out


def summarize_concurrency(rows: list[Row]) -> list[dict[str, Any]]:
    by: dict[tuple[str, int, int], list[Row]] = defaultdict(list)
    for r in rows:
        if r["test"] == "concurrency":
            by[(r["_run"], r["tags"]["rep"], r["tags"]["level"])].append(r)
    levels: dict[int, dict[str, list]] = defaultdict(lambda: {"stream": [], "agg": [],
                                                                "ttft": [], "errors": 0})
    for (_, _, level), group in by.items():
        ok = [r for r in group if r["error"] is None]
        levels[level]["errors"] += len(group) - len(ok)
        levels[level]["stream"] += [decode_tps(r) for r in ok]
        levels[level]["ttft"] += [r["ttft_s"] for r in ok]
        if ok:
            start = min(r["started_at"] for r in ok)
            end = max(r["started_at"] + (r["total_s"] or 0) for r in ok)
            if end > start:
                levels[level]["agg"].append(sum(streamed_tokens(r) for r in ok) / (end - start))
    return [{"level": lv, "stream_tps": _r(_med(v["stream"]), 1),
             "aggregate_tps": _r(_med(v["agg"]), 1), "ttft": _r(_med(v["ttft"]), 3),
             "errors": v["errors"]} for lv, v in sorted(levels.items())]


def summarize(paths: list[Path], include_quick: bool = False) -> dict[str, Any]:
    rows, meta, runs = load_runs(paths, include_quick)
    providers = []
    for name, rs in rows.items():
        m = meta.get(name, {})
        loops = summarize_loops(rs)
        providers.append({
            "name": name, "label": m.get("label", name), "model": m.get("model"),
            "api": m.get("api"), "caching": m.get("caching"), "prices": m.get("prices"),
            "price_source": m.get("price_source"),
            "requests": len(rs), "errors": sum(1 for r in rs if r["error"]),
            "rate_limited": sum(r.get("rate_limited") or 0 for r in rs),
            "spend": _r(sum(r["cost_usd"] or 0 for r in rs)),
            "loops": loops, "long_context": summarize_long_context(rs),
            "concurrency": summarize_concurrency(rs),
        })
    providers.sort(key=lambda p: p["name"])
    all_rows = [r for rs in rows.values() for r in rs]
    return {
        "generated_from": [r["run_id"] for r in runs], "runs": runs,
        "totals": {"requests": len(all_rows), "errors": sum(1 for r in all_rows if r["error"]),
                   "spend": _r(sum(r["cost_usd"] or 0 for r in all_rows)),
                   "input_tokens": sum(r["prompt_tokens"] or 0 for r in all_rows),
                   "providers": len(providers)},
        "providers": providers,
    }


def print_table(s: dict[str, Any]) -> None:
    print(f"\n{s['totals']['requests']} requests, {s['totals']['errors']} errors, "
          f"${s['totals']['spend']:.4f}\n")
    print(f"{'provider':<10} {'50-turn $':>10} {'bug $':>9} {'bug x':>7} {'reuse':>6} "
          f"{'ttft':>6} {'tok/s@1':>8} {'tok/s@8 tot':>11}")
    for p in s["providers"]:
        h, b = p["loops"].get("loop_healthy", {}), p["loops"].get("loop_bug", {})
        c = {x["level"]: x for x in p["concurrency"]}
        fmt = lambda v, f: "-" if v is None else format(v, f)
        print(f"{p['name']:<10} {fmt(h.get('session_cost_mean'), '.4f'):>10} "
              f"{fmt(b.get('session_cost_mean'), '.4f'):>9} "
              f"{fmt(p['loops']['bug_multiplier']['mean'], '.1f'):>7} "
              f"{fmt(h.get('reuse_median'), '.0%'):>6} {fmt(h.get('ttft_median'), '.2f'):>6} "
              f"{fmt(c.get(1, {}).get('stream_tps'), '.0f'):>8} "
              f"{fmt(c.get(8, {}).get('aggregate_tps'), '.0f'):>11}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--include-quick", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("report/summary.json"))
    args = ap.parse_args()
    paths = args.runs or sorted(d for d in Path("results/compare").glob("*") if d.is_dir()
                                and (d / "run.json").exists())
    if not paths:
        sys.exit("no runs found under results/compare/")
    s = summarize(paths, args.include_quick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(s, indent=1))
    print_table(s)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
