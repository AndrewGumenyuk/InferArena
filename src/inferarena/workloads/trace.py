"""Trace-based workload loaders for real request distributions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from inferarena.core.experiment_spec import WorkloadSpec
from inferarena.core.request import Request


def _approximate_tokens(text: str) -> int:
    """Approximate token count from text.

    Uses a simple heuristic (1 token ≈ 0.75 words) suitable when a tokenizer
    is not available. Real traces may override this with pre-computed counts.
    """
    words = len(text.split())
    return max(1, int(words / 0.75))


def _extract_sharegpt_request(item: dict[str, Any]) -> tuple[str, str]:
    """Extract prompt and expected output from a ShareGPT conversation item."""
    conversations = item.get("conversations", [])
    human_parts: list[str] = []
    assistant_parts: list[str] = []
    for turn in conversations:
        role = turn.get("from") or turn.get("role") or ""
        value = turn.get("value") or turn.get("content") or ""
        if role in {"human", "user"}:
            human_parts.append(value)
        elif role in {"gpt", "assistant"}:
            assistant_parts.append(value)
    prompt = "\n".join(human_parts)
    output = assistant_parts[0] if assistant_parts else ""
    return prompt, output


class TraceWorkloadLoader:
    """Load a request trace from JSON/JSONL files.

    Supports the ShareGPT conversation format out of the box and can be
    extended to other formats by mapping fields.
    """

    def __init__(self, spec: WorkloadSpec) -> None:
        """Initialize the loader with a workload spec.

        Args:
            spec: Workload specification. ``trace_path`` must be set.
        """
        if spec.trace_path is None:
            raise ValueError("trace_path is required for trace workloads")
        self.spec = spec
        self.path = Path(spec.trace_path)
        self.rng = random.Random(spec.seed)

    def load(self) -> list[Request]:
        """Load and convert trace records to Request objects."""
        records = self._read_records()
        records = records[: self.spec.num_requests]

        requests: list[Request] = []
        current_time = 0.0
        for record in records:
            prompt, output = self._extract(record)
            inter_arrival = self.rng.expovariate(self.spec.arrival_rate)
            current_time += inter_arrival
            requests.append(
                Request(
                    arrival_time=current_time * 1000.0,
                    prompt_tokens=_approximate_tokens(prompt),
                    max_output_tokens=_approximate_tokens(output)
                    if output
                    else self.spec.output_tokens,
                )
            )
        return requests

    def _read_records(self) -> list[dict[str, Any]]:
        """Read records from JSON or JSONL trace file."""
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported trace structure in {self.path}")

    def _extract(self, record: dict[str, Any]) -> tuple[str, str]:
        """Extract prompt/output text from a trace record."""
        if self.spec.trace_format == "sharegpt":
            return _extract_sharegpt_request(record)
        # Generic flat format: {"prompt": "...", "output": "..."}
        prompt = record.get("prompt") or record.get("input") or ""
        output = record.get("output") or record.get("completion") or ""
        if not isinstance(prompt, str) or not isinstance(output, str):
            raise ValueError(f"Unsupported record format in {self.path}")
        return prompt, output
