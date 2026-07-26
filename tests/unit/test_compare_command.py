"""Tests for the compare command and comparison reporting."""

from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec, WorkloadSpec
from inferarena.reporting.comparison import ComparisonReportGenerator
from inferarena.runner import ExperimentRunner


def test_compare_runs_multiple_schedulers() -> None:
    runner = ExperimentRunner()
    spec = ExperimentSpec(
        name="compare-test",
        workload=WorkloadSpec(
            num_requests=8,
            arrival_rate=2.0,
            prompt_tokens=64,
            output_tokens=8,
            seed=42,
        ),
        engine=EngineSpec(max_tokens_per_step=512),
    )
    results = runner.compare(spec, ["fcfs", "chunked_prefill"])
    assert len(results) == 2
    assert results[0].scheduler_name == "fcfs"
    assert results[1].scheduler_name == "chunked_prefill"


def test_comparison_report_generator() -> None:
    runner = ExperimentRunner()
    spec = ExperimentSpec(
        name="compare-report-test",
        workload=WorkloadSpec(
            num_requests=4,
            arrival_rate=2.0,
            prompt_tokens=32,
            output_tokens=4,
            seed=42,
        ),
        engine=EngineSpec(max_tokens_per_step=512),
    )
    results = runner.compare(spec, ["fcfs", "priority"])
    generator = ComparisonReportGenerator(results)
    table = generator.to_table()
    assert len(table) == 2
    assert table[0]["scheduler"] == "fcfs"
    assert table[1]["scheduler"] == "priority"
    assert "throughput_rps" in table[0]
