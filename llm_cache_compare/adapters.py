"""One streaming call to any supported provider, normalized to the same measurements.

Each provider is called through its own native API (plain HTTP + server-sent events, no SDKs),
so prompt caching behaves the way that provider intends and every usage field is visible:

  openai_chat       POST {base}/chat/completions          (Groq, DeepSeek, fake server)
  openai_responses  POST {base}/responses                 (OpenAI)
  anthropic         POST {base}/v1/messages               (Anthropic; cache_control opt-in)
  gemini            POST {base}/models/{m}:streamGenerateContent?alt=sse   (Google)

Normalized usage: prompt_tokens is ALL input (cached + cache-written + plain), so
"cached / prompt" means the same thing for every provider. Cost is computed from the token
counts with the provider's list prices. Every request goes through the budget guard (hold the
worst case before sending, settle to the computed cost after).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from llm_cache_compare.budget import BudgetGuard
from llm_cache_compare.providers import Provider, cost_from_usage

CHARS_PER_TOKEN_FLOOR = 1.5   # over-counts tokens on purpose, so budget holds are generous
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class CallResult:
    request_id: str
    provider: str
    model: str
    api: str
    test: str
    started_at: float
    tags: dict[str, Any] = field(default_factory=dict)
    request_chars: int = 0
    ttft_s: float | None = None             # first generated token of any kind
    first_content_s: float | None = None    # first visible answer token
    total_s: float | None = None
    last_token_s: float | None = None
    text_chunks: int = 0
    prompt_tokens: int | None = None        # all input tokens
    cached_tokens: int | None = None        # input read from cache
    cache_write_tokens: int | None = None   # input written to cache (if the API says)
    completion_tokens: int | None = None    # output incl. reasoning
    reasoning_tokens: int | None = None
    cost_usd: float | None = None           # computed from tokens x list price
    output_text: str = ""
    finish_reason: str | None = None
    attempts: int = 1
    retry_wait_s: float = 0.0
    rate_limited: int = 0                   # how many 429s we hit on the way
    status_code: int | None = None
    error: str | None = None
    raw_usage: dict[str, Any] | None = None  # exactly what the API reported, for auditing

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def decode_tps(self) -> float | None:
        return decode_tps(self.to_dict())


def streamed_tokens(row: dict[str, Any]) -> int:
    """Output tokens we actually watched arrive. Gemini thinks silently before streaming,
    so its hidden thinking tokens are left out of its speed figure."""
    out = int(row.get("completion_tokens") or 0)
    if row.get("api") == "gemini":
        out -= int(row.get("reasoning_tokens") or 0)
    return max(out, 0)


def decode_tps(row: dict[str, Any]) -> float | None:
    """Output tokens / seconds from first to last streamed token. None if too short to time."""
    first, last = row.get("ttft_s"), row.get("last_token_s")
    tokens = streamed_tokens(row)
    if not tokens or first is None or last is None or (row.get("text_chunks") or 0) < 5:
        return None
    span = last - first
    return tokens / span if span > 0.05 else None


Message = dict[str, Any]   # {"role": "user"|"assistant", "content": str | list[str]}


def _parts(m: Message) -> list[str]:
    c = m["content"]
    return [c] if isinstance(c, str) else list(c)


class APIError(RuntimeError):
    def __init__(self, status: int | None, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _i(x: Any) -> int:
    return int(x or 0)


async def _sse(response: httpx.Response) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
    """Yield (event name, parsed JSON data) for each server-sent event."""
    event: str | None = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                payload = "\n".join(data)
                if payload.strip() != "[DONE]":
                    yield event, json.loads(payload)
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data).strip() != "[DONE]":
        yield event, json.loads("\n".join(data))


class ProviderClient:
    """Streams requests to one provider. One instance per provider per run."""

    def __init__(self, provider: Provider, budget: BudgetGuard, *, timeout_s: float = 300.0,
                 max_attempts: int = 8, base_backoff_s: float = 2.0,
                 max_backoff_s: float = 60.0) -> None:
        key = provider.api_key()
        if not key:
            raise RuntimeError(f"{provider.key_env} is not set (needed for {provider.label})")
        self.p = provider
        self.budget = budget
        self.max_attempts = max_attempts
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30.0),
                                       headers=self._headers(key))
        self._sleep = asyncio.sleep
        self._pace_lock = asyncio.Lock()
        self._last_start = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    def _headers(self, key: str) -> dict[str, str]:
        if self.p.api == "anthropic":
            return {"x-api-key": key, "anthropic-version": "2023-06-01"}
        if self.p.api == "gemini":
            return {"x-goog-api-key": key}
        return {"authorization": f"Bearer {key}"}

    # ---------- request bodies ----------

    def _request(self, system: str, messages: list[Message], max_tokens: int,
                 cache_key: str | None = None) -> tuple[str, dict[str, Any]]:
        """A message's content is a string or a list of parts. Parts let a shared document
        and a changing question be separate blocks, which is how caching sees a "prefix":
        Anthropic caches at explicit breakpoints (we mark every part but the last), and
        OpenAI/Gemini see the same parts as separate content items."""
        p, opts = self.p, dict(self.p.options)
        if p.api == "openai_chat":
            return f"{p.base_url}/chat/completions", {
                "model": p.model, "stream": True, "stream_options": {"include_usage": True},
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             *({"role": m["role"], "content": "\n\n".join(_parts(m))}
                               for m in messages)], **opts}
        if p.api == "openai_responses":
            items = []
            for m in messages:
                parts = _parts(m)
                kind = "input_text" if m["role"] == "user" else "output_text"
                content: list[dict[str, Any]] = [{"type": kind, "text": t} for t in parts]
                if m["role"] == "user":  # like Anthropic: a breakpoint after every part but the
                    for c in content[:-1]:  # last (the implicit one sits at the message end)
                        c["prompt_cache_breakpoint"] = {"mode": "explicit"}
                items.append({"role": m["role"], "content": content})
            body = {"model": p.model, "stream": True, "store": False, "instructions": system,
                    "max_output_tokens": max_tokens, "input": items, **opts}
            if cache_key:
                body["prompt_cache_key"] = cache_key  # routes one conversation to one cache
            return f"{p.base_url}/responses", body
        if p.api == "anthropic":
            msgs = []
            for m in messages:
                parts = _parts(m)
                if len(parts) == 1:
                    msgs.append({"role": m["role"], "content": parts[0]})
                    continue
                blocks: list[dict[str, Any]] = [{"type": "text", "text": t} for t in parts]
                for b in blocks[:-1]:
                    b["cache_control"] = {"type": "ephemeral"}
                msgs.append({"role": m["role"], "content": blocks})
            return f"{p.base_url}/v1/messages", {
                "model": p.model, "stream": True, "max_tokens": max_tokens, "system": system,
                "messages": msgs, **opts}
        if p.api == "gemini":
            gen = dict(opts.pop("generationConfig", {}) or {})
            gen["maxOutputTokens"] = max_tokens
            return f"{p.base_url}/models/{p.model}:streamGenerateContent?alt=sse", {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                              "parts": [{"text": t} for t in _parts(m)]} for m in messages],
                "generationConfig": gen, **opts}
        raise ValueError(p.api)

    # ---------- public call ----------

    async def call(self, system: str, messages: list[Message], *, max_tokens: int,
                   test: str, tags: dict[str, Any] | None = None,
                   cache_key: str | None = None) -> CallResult:
        url, body = self._request(system, messages, max_tokens, cache_key)
        chars = len(json.dumps(body))
        result = CallResult(request_id=uuid.uuid4().hex, provider=self.p.name,
                            model=self.p.model, api=self.p.api, test=test,
                            started_at=time.time(), tags=dict(tags or {}), request_chars=chars)
        worst = self.p.worst_prices
        hold = await self.budget.reserve(
            ((chars / CHARS_PER_TOKEN_FLOOR + 1) * worst.dearest_input
             + max_tokens * worst.output) / 1e6)
        try:
            for n in range(1, self.max_attempts + 1):
                result.attempts = n
                try:
                    await self._stream(url, body, result)
                    result.error = result.status_code = None
                    break
                except (APIError, httpx.TransportError) as e:
                    status = getattr(e, "status", None)
                    result.status_code = status
                    result.error = f"{type(e).__name__}: {str(e)[:300]}"
                    if status == 429:
                        result.rate_limited += 1
                    retryable = status in RETRY_STATUS or isinstance(e, httpx.TransportError)
                    if not retryable or n == self.max_attempts:
                        break
                    wait = getattr(e, "retry_after", None)
                    if wait is None:
                        wait = random.random() * min(self.max_backoff_s,
                                                     self.base_backoff_s * 2 ** (n - 1))
                    wait = min(wait, self.max_backoff_s)
                    result.retry_wait_s += wait
                    _reset_measurements(result)
                    await self._sleep(wait)
        finally:
            # Rejected before streaming (HTTP error) -> nothing billed. Failed mid-stream with
            # unknown cost -> charge the full hold so the budget errs on the safe side.
            if result.cost_usd is not None:
                charged = result.cost_usd
            elif result.status_code is not None and result.ttft_s is None:
                charged = 0.0
            else:
                charged = hold
            await self.budget.settle(hold, charged)
        return result

    async def _wait_turn(self) -> None:
        """Space request starts at least min_interval_s apart (strict free-tier limits)."""
        if self.p.min_interval_s <= 0:
            return
        async with self._pace_lock:
            wait = self._last_start + self.p.min_interval_s - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._last_start = time.monotonic()

    async def _stream(self, url: str, body: dict[str, Any], result: CallResult) -> None:
        await self._wait_turn()
        t0 = time.perf_counter()
        async with self._http.stream("POST", url, json=body) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode(errors="replace")
                ra = resp.headers.get("retry-after")
                try:
                    retry_after = float(ra) if ra else None
                except ValueError:
                    retry_after = None
                if retry_after is None:  # Gemini puts it in the body: "retryDelay": "37s"
                    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', text)
                    retry_after = float(m.group(1)) + 1.0 if m else None
                raise APIError(resp.status_code, f"HTTP {resp.status_code}: {text[:300]}",
                               retry_after)
            parser = getattr(self, f"_parse_{self.p.api}")
            parts: list[str] = []
            last_at: float | None = None
            async for event, data in _sse(resp):
                now = time.perf_counter() - t0
                if (err := _stream_error(event, data)) is not None:
                    raise err
                text, thinking = parser(data, result, event)
                if text or thinking:
                    if result.ttft_s is None:
                        result.ttft_s = now
                    result.text_chunks += 1
                    last_at = now
                if text:
                    if result.first_content_s is None:
                        result.first_content_s = now
                    parts.append(text)
        result.total_s = time.perf_counter() - t0
        result.last_token_s = last_at
        result.output_text = "".join(parts)
        if result.prompt_tokens is None:
            raise APIError(None, "stream ended without usage")
        result.cost_usd = cost_from_usage(
            self.p, prompt_tokens=result.prompt_tokens, cached_tokens=result.cached_tokens or 0,
            cache_write_tokens=result.cache_write_tokens or 0,
            completion_tokens=result.completion_tokens or 0)

    # ---------- per-API stream parsers: return (answer text, thinking text) ----------

    @staticmethod
    def _parse_openai_chat(d: dict[str, Any], r: CallResult, _e: str | None) -> tuple[str, str]:
        text = thinking = ""
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            text += delta.get("content") or ""
            thinking += delta.get("reasoning_content") or delta.get("reasoning") or ""
            if ch.get("finish_reason"):
                r.finish_reason = ch["finish_reason"]
        usage = d.get("usage") or (d.get("x_groq") or {}).get("usage")
        if usage:
            pd = usage.get("prompt_tokens_details") or {}
            cd = usage.get("completion_tokens_details") or {}
            r.raw_usage = usage
            r.prompt_tokens = _i(usage.get("prompt_tokens"))
            # DeepSeek reports prompt_cache_hit_tokens instead of prompt_tokens_details
            r.cached_tokens = _i(pd.get("cached_tokens") or usage.get("prompt_cache_hit_tokens"))
            r.completion_tokens = _i(usage.get("completion_tokens"))
            r.reasoning_tokens = cd.get("reasoning_tokens")
        return text, thinking

    @staticmethod
    def _parse_openai_responses(d: dict[str, Any], r: CallResult,
                                e: str | None) -> tuple[str, str]:
        etype = d.get("type") or e or ""
        if etype == "response.output_text.delta":
            return d.get("delta") or "", ""
        if etype.startswith("response.reasoning") and etype.endswith(".delta"):
            return "", d.get("delta") or ""
        if etype in ("response.completed", "response.incomplete"):
            resp = d.get("response") or {}
            usage = resp.get("usage") or {}
            idt = usage.get("input_tokens_details") or {}
            odt = usage.get("output_tokens_details") or {}
            r.raw_usage = usage
            r.prompt_tokens = _i(usage.get("input_tokens"))
            r.cached_tokens = _i(idt.get("cached_tokens"))
            r.cache_write_tokens = _i(idt.get("cache_write_tokens"))
            r.completion_tokens = _i(usage.get("output_tokens"))
            r.reasoning_tokens = odt.get("reasoning_tokens")
            r.finish_reason = resp.get("status")
        return "", ""

    @staticmethod
    def _parse_anthropic(d: dict[str, Any], r: CallResult, e: str | None) -> tuple[str, str]:
        etype = d.get("type") or e or ""
        usage = None
        if etype == "message_start":
            usage = (d.get("message") or {}).get("usage")
        elif etype == "message_delta":
            usage = d.get("usage")
            r.finish_reason = (d.get("delta") or {}).get("stop_reason") or r.finish_reason
        elif etype == "content_block_delta":
            delta = d.get("delta") or {}
            if delta.get("type") == "text_delta":
                return delta.get("text") or "", ""
            if delta.get("type") == "thinking_delta":
                return "", delta.get("thinking") or ""
        if usage:
            merged = {**(r.raw_usage or {}), **{k: v for k, v in usage.items() if v is not None}}
            r.raw_usage = merged
            plain = _i(merged.get("input_tokens"))
            read = _i(merged.get("cache_read_input_tokens"))
            write = _i(merged.get("cache_creation_input_tokens"))
            r.prompt_tokens = plain + read + write
            r.cached_tokens = read
            r.cache_write_tokens = write
            r.completion_tokens = _i(merged.get("output_tokens"))
        return "", ""

    @staticmethod
    def _parse_gemini(d: dict[str, Any], r: CallResult, _e: str | None) -> tuple[str, str]:
        text = thinking = ""
        for cand in d.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if part.get("thought"):
                    thinking += part.get("text") or ""
                else:
                    text += part.get("text") or ""
            if cand.get("finishReason"):
                r.finish_reason = cand["finishReason"]
        um = d.get("usageMetadata")
        if um and um.get("promptTokenCount") is not None:
            r.raw_usage = um
            r.prompt_tokens = _i(um.get("promptTokenCount"))
            r.cached_tokens = _i(um.get("cachedContentTokenCount"))
            thoughts = _i(um.get("thoughtsTokenCount"))
            r.completion_tokens = _i(um.get("candidatesTokenCount")) + thoughts
            r.reasoning_tokens = thoughts
        return text, thinking


def _stream_error(event: str | None, data: dict[str, Any]) -> APIError | None:
    """Errors some APIs send inside an otherwise-200 stream."""
    etype = data.get("type") or event or ""
    if etype in ("error", "response.failed") or ("error" in data and len(data) <= 2):
        err = data.get("error") or (data.get("response") or {}).get("error") or data
        msg = err.get("message") if isinstance(err, dict) else str(err)
        kind = (err.get("type") or err.get("code") or "") if isinstance(err, dict) else ""
        kind_s = str(kind)
        status = (402 if "quota" in kind_s or "billing" in kind_s else  # out of credit: stop
                  529 if "overloaded" in kind_s else 429 if "rate" in kind_s else 500)
        return APIError(status, f"stream error: {kind} {msg}"[:300])
    return None


def _reset_measurements(r: CallResult) -> None:
    for f in ("ttft_s", "first_content_s", "total_s", "last_token_s", "prompt_tokens",
              "cached_tokens", "cache_write_tokens", "completion_tokens", "reasoning_tokens",
              "cost_usd", "finish_reason", "raw_usage"):
        setattr(r, f, None)
    r.text_chunks = 0
    r.output_text = ""
