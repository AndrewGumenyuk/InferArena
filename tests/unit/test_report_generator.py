"""Tests for the report generator."""

from pathlib import Path

from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec, WorkloadSpec
from inferarena.metrics.collector import MetricsCollector
from inferarena.metrics.result import ExperimentResult, RequestResult
from inferarena.reporting.generator import ReportGenerator


def test_report_generator_saves_requests(tmp_path: Path) -> None:
    result = ExperimentResult(
        scheduler_name="fcfs",
        total_steps=10,
        total_time=100.0,
        completed_requests=1,
        request_results=[
            RequestResult(
                request_id="r1",
                arrival_time=0.0,
                prompt_tokens=8,
                max_output_tokens=4,
                scheduled_time=1.0,
                first_token_time=2.0,
                completion_time=10.0,
            ),
        ],
    )
    metrics = MetricsCollector()
    ReportGenerator(result, metrics).save(tmp_path)

    requests_path = tmp_path / "requests.json"
    assert requests_path.exists()
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    telemetry_path = tmp_path / "telemetry.jsonl"
    assert telemetry_path.exists()


def test_report_generator_end_to_end(tmp_path: Path) -> None:
    from inferarena.runner import ExperimentRunner

    runner = ExperimentRunner()
    spec = ExperimentSpec(
        name="report-test",
        workload=WorkloadSpec(
            num_requests=4,
            arrival_rate=2.0,
            prompt_tokens=32,
            output_tokens=4,
            seed=42,
        ),
        engine=EngineSpec(max_tokens_per_step=512),
        output_dir=tmp_path,
    )
    result = runner.run(spec)
    assert result.completed_requests == 4
    assert (tmp_path / "requests.json").exists()
    assert (tmp_path / "summary.json").exists()
