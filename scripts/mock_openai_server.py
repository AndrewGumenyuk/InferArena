#!/usr/bin/env python3
"""OpenAI-compatible mock server for InferArena demos.

This server mimics the API surface of vLLM/SGLang/TensorRT-LLM without
needing a GPU. It is useful for demonstrating InferArena's real-engine
integration on any machine. Do not use it for performance benchmarking.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token count from message text."""
    text = "\n".join(str(m.get("content", "")) for m in messages)
    # Very rough heuristic: ~4 characters per token.
    return max(1, len(text) // 4)


class Handler(BaseHTTPRequestHandler):
    """Handle OpenAI-compatible chat completion requests."""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
        """Suppress default request logging."""
        pass

    def _json_response(self, status: int, data: dict[str, Any]) -> None:  # noqa: ANN401
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json_response(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode())
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid json"})
            return

        messages = payload.get("messages", [])
        prompt_tokens = estimate_tokens(messages)
        max_tokens = payload.get("max_tokens", 32)
        stream = payload.get("stream", False)
        model = payload.get("model", "mock-model")

        # Fake timing calibrated to feel plausible, not realistic.
        ttft_ms = 10 + prompt_tokens * 0.5
        tbt_ms = 15

        if not stream:
            content = " ".join(f"token{i}" for i in range(max_tokens))
            self._json_response(
                200,
                {
                    "id": "mock-completion",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": prompt_tokens + max_tokens,
                    },
                },
            )
            return

        # Streaming response.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        time.sleep(ttft_ms / 1000.0)

        for i in range(max_tokens):
            chunk = {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"token{i} "},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(tbt_ms / 1000.0)

        done_chunk = {
            "id": "mock-completion",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        self._json_response(200, {"object": "list", "data": []})


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Mock OpenAI server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
