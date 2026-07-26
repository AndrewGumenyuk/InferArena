"""Tests for the notebook dashboard helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferarena.reporting.dashboard import (
    discover_experiments,
    load_request_results,
    to_dataframe,
)


def test_discover_experiments_finds_summaries(tmp_path: Path) -> None:
    output_dir = tmp_path / "inferarena_outputs" / "fcfs"
    output_dir.mkdir(parents=True)
    summary = {"scheduler": "fcfs", "throughput_rps": 1.23}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f)

    found = discover_experiments(tmp_path / "inferarena_outputs")
    assert len(found) == 1
    assert found[0]["scheduler"] == "fcfs"
    assert "path" in found[0]


def test_discover_experiments_returns_empty_for_missing_dir() -> None:
    found = discover_experiments("nonexistent_outputs")
    assert found == []


def test_load_request_results(tmp_path: Path) -> None:
    summary = {"path": str(tmp_path)}
    requests = [{"request_id": "r1", "arrival_time": 0.0, "completion_time": 100.0}]
    with (tmp_path / "requests.json").open("w", encoding="utf-8") as f:
        json.dump(requests, f)

    loaded = load_request_results(summary)
    assert loaded == requests


def test_load_request_results_missing_file() -> None:
    summary = {"path": "/nonexistent/path"}
    assert load_request_results(summary) == []


def test_to_dataframe() -> None:
    summaries = [{"scheduler": "fcfs", "throughput_rps": 1.0}]
    df = to_dataframe(summaries)
    assert len(df) == 1
    assert df["scheduler"][0] == "fcfs"


def test_plot_latency_cdf_requires_matplotlib() -> None:
    pytest.importorskip("matplotlib", reason="matplotlib not installed")
    from inferarena.reporting.dashboard import plot_latency_cdf

    summaries = [
        {
            "scheduler": "fcfs",
            "path": "/tmp",
        }
    ]
    ax = plot_latency_cdf(summaries)
    assert ax is not None
