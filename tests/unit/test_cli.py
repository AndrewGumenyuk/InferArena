"""Tests for the InferArena CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from inferarena.cli.main import app

runner = CliRunner()


def test_run_command(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "name: cli-test\n"
        "scheduler: fcfs\n"
        "cache_policy: no_op\n"
        "workload:\n"
        "  name: uniform\n"
        "  num_requests: 4\n"
        "  arrival_rate: 2.0\n"
        "  prompt_tokens: 32\n"
        "  output_tokens: 4\n"
        "  seed: 42\n"
        "engine:\n"
        "  name: simulation\n"
        "  max_tokens_per_step: 512\n"
        f"output_dir: {tmp_path / 'outputs'}\n"
    )
    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "fcfs" in result.output
    assert "Completed" in result.output


def test_list_schedulers_command() -> None:
    result = runner.invoke(app, ["list-schedulers"])
    assert result.exit_code == 0, result.output
    assert "fcfs" in result.output


def test_list_cache_policies_command() -> None:
    result = runner.invoke(app, ["list-cache-policies"])
    assert result.exit_code == 0, result.output
    assert "no_op" in result.output


def test_list_routers_command() -> None:
    result = runner.invoke(app, ["list-routers"])
    assert result.exit_code == 0, result.output
    assert "round_robin" in result.output


def test_download_dataset_command(tmp_path: Path) -> None:
    with patch("inferarena.cli.main.download_dataset") as mock_download:
        mock_download.return_value = tmp_path / "sharegpt.json"
        result = runner.invoke(
            app,
            ["download-dataset", "sharegpt_vicuna", "--output", str(tmp_path / "sharegpt.json")],
        )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with("sharegpt_vicuna", tmp_path / "sharegpt.json")


def test_dashboard_command_missing_streamlit(tmp_path: Path) -> None:
    with patch.dict("sys.modules", {"streamlit": None}):
        result = runner.invoke(app, ["dashboard", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "streamlit" in result.output.lower()
