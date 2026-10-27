"""Synthetic agent workloads: system prompts and tool outputs of a target token size.

Content is deterministic (no randomness) so runs are reproducible. Each run gets a unique
nonce at the very start of the system prompt, so it never reuses another run's cache.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

# Characters per token for this generator's text on Coral's tokenizer.
# Calibrated 2026-10-05: 3.163 chars/token, ~0 fixed overhead (scripts/calibrate_tokens.py).
DEFAULT_CHARS_PER_TOKEN = 3.16

TEMPLATES = (
    "PASS tests/test_module_{a}.py::test_case_{b} ({c}.{d}s)",
    "src/service_{a}/handler.py:{b}: warning: unused variable 'value_{c}'",
    "def helper_{a}(items, limit={b}):\n    return [x for x in items if x.score > {c}]",
    "INFO request id={a}-{b} path=/api/v{c}/items status=200 latency_ms={d}",
    "git diff: modified src/component_{a}.ts (+{b} -{c} lines)",
)


def new_run_nonce() -> str:
    return uuid.uuid4().hex


def system_prompt(nonce: str) -> str:
    return (
        f"Run {nonce}. You are a coding agent working in a large repository. "
        "After each tool output, reply with one short sentence describing the next action."
    )


@dataclass
class ToolOutputGenerator:
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    def make(self, target_tokens: int, step: int) -> str:
        """Deterministic tool output of roughly target_tokens tokens for a given step."""
        if target_tokens <= 0:
            raise ValueError("target_tokens must be > 0")
        target_chars = int(target_tokens * self.chars_per_token)
        header = f"$ tool call #{step}"
        lines = [header]
        size = len(header)
        i = 0
        while size < target_chars:
            template = TEMPLATES[(step + i) % len(TEMPLATES)]
            line = template.format(
                a=(step * 31 + i) % 97, b=(step * 17 + i * 7) % 503,
                c=(i * 13 + step) % 89, d=(i * 7 + step * 3) % 997,
            )
            lines.append(line)
            size += len(line) + 1
            i += 1
        return "\n".join(lines)
