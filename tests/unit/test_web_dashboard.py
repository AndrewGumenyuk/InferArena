"""Tests for the Streamlit web dashboard helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from inferarena.web.app import _plot_latency_cdf, _summaries_to_df

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")


def test_summaries_to_df(tmp_path: Path) -> None:
    output_dir = tmp_path / "inferarena_outputs" / "fcfs"
    output_dir.mkdir(parents=True)
    summary = {"scheduler": "fcfs", "throughput_rps": 1.23}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f)

    from inferarena.reporting.dashboard import discover_experiments

    summaries = discover_experiments(tmp_path / "inferarena_outputs")
    df = _summaries_to_df(summaries)
    assert len(df) == 1
    assert df["scheduler"][0] == "fcfs"


def test_plot_latency_cdf(tmp_path: Path) -> None:
    output_dir = tmp_path / "inferarena_outputs" / "fcfs"
    output_dir.mkdir(parents=True)
    summary = {"scheduler": "fcfs", "throughput_rps": 1.23}
    requests = [
        {"arrival_time": 0.0, "completion_time": 100.0},
        {"arrival_time": 10.0, "completion_time": 150.0},
    ]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f)
    with (output_dir / "requests.json").open("w", encoding="utf-8") as f:
        json.dump(requests, f)

    summary["path"] = str(output_dir)
    fig = _plot_latency_cdf([summary])
    assert fig is not None


def test_dashboard_cli_invokes_streamlit(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from inferarena.cli.main import app

    runner = CliRunner()
    with patch("inferarena.cli.main.run_dashboard") as mock_run:
        result = runner.invoke(app, ["dashboard", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once_with(tmp_path)
