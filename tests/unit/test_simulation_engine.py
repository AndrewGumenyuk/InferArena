"""Tests for the simulation engine."""

from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec, WorkloadSpec
from inferarena.core.request import Request
from inferarena.plugins.schedulers.fcfs import FCFSScheduler
from inferarena.simulation.engine import SimulationEngine


def test_simulation_completes_all_requests() -> None:
    scheduler = FCFSScheduler()
    engine_spec = EngineSpec(max_tokens_per_step=512)
    engine = SimulationEngine(scheduler, engine_spec)
    spec = ExperimentSpec(
        workload=WorkloadSpec(
            num_requests=4,
            arrival_rate=1.0,
            prompt_tokens=64,
            output_tokens=16,
            seed=42,
        ),
        engine=engine_spec,
    )
    result = engine.run(spec)
    assert result.completed_requests == spec.workload.num_requests
    assert result.total_steps > 0
    assert result.throughput > 0


def test_simulation_summary_has_expected_keys() -> None:
    scheduler = FCFSScheduler()
    engine = SimulationEngine(scheduler)
    spec = ExperimentSpec(
        workload=WorkloadSpec(
            num_requests=2,
            arrival_rate=1.0,
            prompt_tokens=32,
            output_tokens=8,
            seed=42,
        ),
    )
    result = engine.run(spec)
    summary = result.summary()
    assert "scheduler" in summary
    assert "completed_requests" in summary
    assert "throughput_rps" in summary
    assert "ttft_p50_ms" in summary
    assert "ttft_p99_ms" in summary
    assert "latency_p50_ms" in summary
    assert "latency_p99_ms" in summary
    assert "queue_time_p50_ms" in summary
    assert "tbt_p50_ms" in summary
    assert "prefill_p50_ms" in summary


def test_simulation_fast_forwards_past_idle_time() -> None:
    scheduler = FCFSScheduler()
    engine_spec = EngineSpec(max_tokens_per_step=512)
    engine = SimulationEngine(scheduler, engine_spec)
    requests = [
        Request(
            arrival_time=1000.0,
            prompt_tokens=10,
            max_output_tokens=2,
            request_id="late-request",
        ),
    ]
    result = engine._run_requests(requests, max_steps=10000)
    assert result.completed_requests == 1
    # Without fast-forward the engine would take ~1000 empty 1 ms steps.
    # With fast-forward it should be just prefill + 2 decode steps.
    assert result.total_steps == 3
    assert result.total_time >= 1000.0
