"""Tests for the benchmark command."""

from inferarena.core.experiment_spec import ExperimentSpec, WorkloadSpec
from inferarena.runner import ExperimentRunner


def test_benchmark_runs_multiple_configs(tmp_path) -> None:
    base = tmp_path / "benchmarks"
    base.mkdir()

    spec_a = ExperimentSpec(
        name="a",
        workload=WorkloadSpec(num_requests=4, arrival_rate=2.0, prompt_tokens=32, output_tokens=4),
    )
    spec_b = ExperimentSpec(
        name="b",
        workload=WorkloadSpec(num_requests=4, arrival_rate=2.0, prompt_tokens=32, output_tokens=4),
    )
    spec_a.output_dir = base / "a"
    spec_b.output_dir = base / "b"

    # Pydantic v2 supports json dump; for YAML we reuse the model dict.
    import yaml

    for name, spec in [("a", spec_a), ("b", spec_b)]:
        path = base / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(spec.model_dump(mode="json"), f)

    runner = ExperimentRunner()
    results = runner.benchmark([spec_a, spec_b])
    assert len(results) == 2
    assert all(r.completed_requests == 4 for r in results)
