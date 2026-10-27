"""Hard spending limits: refuse a request before it could push spend past a per-run or all-time cap.

Works like a hotel card hold: reserve() places a hold for the worst-case cost of a request,
settle() replaces the hold with the real cost reported by the API. Holds make it safe
when several requests run at once.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """Raised when a request would push spend past a cap."""


@dataclass
class BudgetGuard:
    run_cap_usd: float = 5.0
    total_cap_usd: float = 5.0
    ledger_path: Path = Path("results/spend_ledger.json")
    run_spent: float = 0.0
    _reserved: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.ledger_path = Path(self.ledger_path)
        self._total_before_run = self._read_total()

    def _read_total(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        return float(json.loads(self.ledger_path.read_text())["total_spent_usd"])

    def _write_total(self, total: float) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"total_spent_usd": round(total, 8)}))
        tmp.replace(self.ledger_path)  # atomic: never leaves a half-written ledger

    @property
    def total_spent(self) -> float:
        return self._total_before_run + self.run_spent

    async def reserve(self, estimate_usd: float) -> float:
        """Place a hold for a request's worst-case cost. Raises BudgetExceeded if it won't fit."""
        if estimate_usd < 0:
            raise ValueError("estimate must be >= 0")
        async with self._lock:
            committed = self.run_spent + self._reserved + estimate_usd
            if committed > self.run_cap_usd:
                raise BudgetExceeded(
                    f"run cap ${self.run_cap_usd:.2f} would be exceeded "
                    f"(spent ${self.run_spent:.4f}, held ${self._reserved:.4f}, "
                    f"next up to ${estimate_usd:.4f})"
                )
            if self._total_before_run + committed > self.total_cap_usd:
                raise BudgetExceeded(
                    f"total cap ${self.total_cap_usd:.2f} would be exceeded "
                    f"(all-time ${self.total_spent:.4f}, next up to ${estimate_usd:.4f})"
                )
            self._reserved += estimate_usd
            return estimate_usd

    async def settle(self, reserved_usd: float, actual_usd: float) -> None:
        """Release the hold and record the real cost (0 if the request failed with no charge)."""
        async with self._lock:
            self._reserved -= reserved_usd
            self.run_spent += actual_usd
            self._write_total(self.total_spent)
