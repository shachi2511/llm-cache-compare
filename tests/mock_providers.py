"""Fake versions of all four provider APIs, for tests: no network, no keys, no spend.

Routes (any path prefix is accepted):
  .../chat/completions                    OpenAI-compatible chat (Groq, DeepSeek)
  .../responses                           OpenAI Responses
  .../v1/messages                         Anthropic Messages (caches only with cache_control)
  .../models/<model>:streamGenerateContent  Gemini

Each fake streams a few text chunks, then usage in that API's own field names, and imitates
prefix caching: cached tokens = longest prefix shared with any earlier prompt, in 64-token
chunks, once the prompt is at least 256 tokens. `fail_first` makes the first N requests
return 429 so retry handling can be tested.

Run standalone: PYTHONPATH=. python tests/mock_providers.py --port 8199
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CHARS_PER_TOKEN = 3
CHUNK = 64
MIN_CACHE = 256
REPLY = "Next I will run the failing test with verbose output."


class State:
    def __init__(self, fail_first: int = 0, delay_s: float = 0.002) -> None:
        self.lock = threading.Lock()
        self.seen: list[str] = []
        self.fail_first = fail_first
        self.delay_s = delay_s
        self.requests = 0
        self.bodies: list[dict[str, Any]] = []

    def cache(self, prompt: str) -> int:
        """Tokens of `prompt` already cached (longest shared prefix with an earlier prompt)."""
        with self.lock:
            best = 0
            for old in self.seen:
                n = 0
                limit = min(len(old), len(prompt))
                while n < limit and old[n] == prompt[n]:
                    n += 1
                best = max(best, n)
            self.seen.append(prompt)
        tokens = best // CHARS_PER_TOKEN
        return 0 if tokens < MIN_CACHE else tokens // CHUNK * CHUNK


def _tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _text(content: Any) -> str:
    """A message's content: a string, or a list of text blocks/parts."""
    if isinstance(content, str):
        return content
    return "\x01".join(b.get("text", "") for b in content)


def _prompt_text(kind: str, body: dict[str, Any]) -> str:
    if kind == "chat":
        return "\x00".join(_text(m["content"]) for m in body["messages"])
    if kind == "responses":
        return "\x00".join([body.get("instructions") or ""]
                           + [_text(m["content"]) for m in body["input"]])
    if kind == "anthropic":
        return "\x00".join([body.get("system") or ""]
                           + [_text(m["content"]) for m in body["messages"]])
    sys_text = "".join(p["text"] for p in body["systemInstruction"]["parts"])
    return "\x00".join([sys_text] + [_text(c["parts"]) for c in body["contents"]])


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # quiet
            pass

        def _sse(self, data: dict[str, Any], event: str | None = None) -> None:
            chunk = (f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()
            time.sleep(state.delay_s)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            path = self.path.split("?")[0]
            kind = ("chat" if path.endswith("/chat/completions") else
                    "responses" if path.endswith("/responses") else
                    "anthropic" if path.endswith("/v1/messages") else
                    "gemini" if ":streamGenerateContent" in path else None)
            with state.lock:
                state.requests += 1
                n = state.requests
                state.bodies.append(body)
            if kind is None:
                self.send_error(404)
                return
            if n <= state.fail_first:
                payload = b'{"error": {"message": "rate limited", "type": "rate_limit"}}'
                self.send_response(429)
                self.send_header("retry-after", "0")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if not (self.headers.get("authorization") or self.headers.get("x-api-key")
                    or self.headers.get("x-goog-api-key")):
                self.send_error(401)
                return
            prompt = _prompt_text(kind, body)
            total = _tokens(prompt)
            cached = state.cache(prompt)
            words = REPLY.split(" ")
            out = len(words) * 2
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            getattr(self, f"_{kind}")(body, words, total, cached, out)

        def _chat(self, body, words, total, cached, out) -> None:
            self._sse({"choices": [{"delta": {"reasoning": "thinking..."}}]})
            for w in words:
                self._sse({"choices": [{"delta": {"content": w + " "}}]})
            self._sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            self._sse({"choices": [], "usage": {
                "prompt_tokens": total, "completion_tokens": out,
                "prompt_tokens_details": {"cached_tokens": cached},
                "completion_tokens_details": {"reasoning_tokens": 3}}})
            self.wfile.write(b"data: [DONE]\n\n")

        def _responses(self, body, words, total, cached, out) -> None:
            self._sse({"type": "response.created"}, "response.created")
            for w in words:
                self._sse({"type": "response.output_text.delta", "delta": w + " "},
                          "response.output_text.delta")
            self._sse({"type": "response.completed", "response": {
                "status": "completed", "usage": {
                    "input_tokens": total, "output_tokens": out,
                    "input_tokens_details": {"cached_tokens": cached,
                                             "cache_write_tokens": total - cached},
                    "output_tokens_details": {"reasoning_tokens": 0}}}},
                "response.completed")

        def _anthropic(self, body, words, total, cached, out) -> None:
            caching = "cache_control" in body or "cache_control" in json.dumps(body["messages"])
            read = cached if caching else 0
            write = (total - cached) if caching and total >= MIN_CACHE else 0
            plain = total - read - write
            self._sse({"type": "message_start", "message": {"usage": {
                "input_tokens": plain, "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write, "output_tokens": 1}}}, "message_start")
            for w in words:
                self._sse({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": w + " "}},
                          "content_block_delta")
            self._sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": out}}, "message_delta")
            self._sse({"type": "message_stop"}, "message_stop")

        def _gemini(self, body, words, total, cached, out) -> None:
            for i, w in enumerate(words):
                chunk: dict[str, Any] = {"candidates": [{"content": {
                    "role": "model", "parts": [{"text": w + " "}]}}]}
                if i == len(words) - 1:
                    chunk["candidates"][0]["finishReason"] = "STOP"
                    chunk["usageMetadata"] = {
                        "promptTokenCount": total, "cachedContentTokenCount": cached,
                        "candidatesTokenCount": out, "thoughtsTokenCount": 5}
                self._sse(chunk)

    return Handler


def serve(port: int = 0, **kw: Any) -> tuple[ThreadingHTTPServer, State]:
    state = State(**kw)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    srv, _ = serve(ap.parse_args().port)
    print(f"fake providers on http://127.0.0.1:{srv.server_port}")
    threading.Event().wait()
