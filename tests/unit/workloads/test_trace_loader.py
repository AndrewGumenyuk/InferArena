"""Tests for the trace workload loader."""

import json

import pytest

from inferarena.core.experiment_spec import WorkloadSpec
from inferarena.workloads.trace import TraceWorkloadLoader


def test_load_sharegpt_jsonl(tmp_path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    records = [
        {
            "id": "1",
            "conversations": [
                {"from": "human", "value": "Hello world example"},
                {"from": "gpt", "value": "print('hello world')"},
            ],
        },
        {
            "id": "2",
            "conversations": [
                {"from": "human", "value": "What is two plus two?"},
                {"from": "gpt", "value": "Four."},
            ],
        },
    ]
    trace_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    spec = WorkloadSpec(
        name="trace",
        trace_path=str(trace_file),
        trace_format="sharegpt",
        num_requests=10,
        arrival_rate=1.0,
        seed=42,
    )
    requests = TraceWorkloadLoader(spec).load()
    assert len(requests) == 2
    assert all(r.prompt_tokens > 0 for r in requests)
    assert all(r.max_output_tokens > 0 for r in requests)
    assert requests[0].arrival_time <= requests[1].arrival_time


def test_load_generic_json(tmp_path) -> None:
    trace_file = tmp_path / "trace.json"
    record = {"prompt": "Short prompt", "output": "Short output"}
    trace_file.write_text(json.dumps(record), encoding="utf-8")

    spec = WorkloadSpec(
        name="trace",
        trace_path=str(trace_file),
        trace_format="generic",
        num_requests=1,
        arrival_rate=1.0,
        seed=42,
    )
    requests = TraceWorkloadLoader(spec).load()
    assert len(requests) == 1
    assert requests[0].prompt_tokens > 0
    assert requests[0].max_output_tokens > 0


def test_truncates_to_num_requests(tmp_path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    records = [{"prompt": f"prompt {i}", "output": f"output {i}"} for i in range(10)]
    trace_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    spec = WorkloadSpec(
        name="trace",
        trace_path=str(trace_file),
        trace_format="generic",
        num_requests=3,
        arrival_rate=1.0,
        seed=42,
    )
    requests = TraceWorkloadLoader(spec).load()
    assert len(requests) == 3


def test_missing_trace_path_raises() -> None:
    spec = WorkloadSpec(name="trace", trace_path=None)
    with pytest.raises(ValueError, match="trace_path is required"):
        TraceWorkloadLoader(spec)


def test_empty_output_uses_spec_default(tmp_path) -> None:
    trace_file = tmp_path / "trace.json"
    record = {"prompt": "Just a prompt", "output": ""}
    trace_file.write_text(json.dumps(record), encoding="utf-8")

    spec = WorkloadSpec(
        name="trace",
        trace_path=str(trace_file),
        trace_format="generic",
        num_requests=1,
        arrival_rate=1.0,
        output_tokens=64,
        seed=42,
    )
    requests = TraceWorkloadLoader(spec).load()
    assert len(requests) == 1
    assert requests[0].max_output_tokens == 64
