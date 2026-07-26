"""Tests for the multi-GPU simulation engine."""

from inferarena.core.experiment_spec import (
    ClusterSpec,
    EngineSpec,
    ExperimentSpec,
    WorkerSpec,
    WorkloadSpec,
)
from inferarena.plugins.schedulers.fcfs import FCFSScheduler
from inferarena.simulation.multi_gpu_engine import MultiGPUSimulationEngine


def test_multi_gpu_engine_completes_all_requests() -> None:
    scheduler = FCFSScheduler()
    engine = MultiGPUSimulationEngine(scheduler, EngineSpec(max_tokens_per_step=512))
    spec = ExperimentSpec(
        engine=EngineSpec(name="multi_gpu_simulation", max_tokens_per_step=512),
        cluster=ClusterSpec(
            workers=[WorkerSpec(gpu_count=1), WorkerSpec(gpu_count=1)],
            router="round_robin",
        ),
        workload=WorkloadSpec(
            num_requests=16,
            arrival_rate=2.0,
            prompt_tokens=64,
            output_tokens=8,
            seed=42,
        ),
    )
    result = engine.run(spec)
    assert result.completed_requests == spec.workload.num_requests
    assert result.throughput > 0


def test_least_loaded_router_balances_load() -> None:
    scheduler = FCFSScheduler()
    engine = MultiGPUSimulationEngine(scheduler, EngineSpec(max_tokens_per_step=512))
    spec = ExperimentSpec(
        engine=EngineSpec(name="multi_gpu_simulation", max_tokens_per_step=512),
        cluster=ClusterSpec(
            workers=[WorkerSpec(gpu_count=1), WorkerSpec(gpu_count=1)],
            router="least_loaded",
        ),
        workload=WorkloadSpec(
            num_requests=8,
            arrival_rate=2.0,
            prompt_tokens=32,
            output_tokens=4,
            seed=42,
        ),
    )
    result = engine.run(spec)
    assert result.completed_requests == spec.workload.num_requests


def test_multi_gpu_aggregates_request_results() -> None:
    scheduler = FCFSScheduler()
    engine = MultiGPUSimulationEngine(scheduler, EngineSpec(max_tokens_per_step=512))
    spec = ExperimentSpec(
        engine=EngineSpec(name="multi_gpu_simulation", max_tokens_per_step=512),
        cluster=ClusterSpec(
            workers=[WorkerSpec(gpu_count=1), WorkerSpec(gpu_count=1)],
            router="round_robin",
        ),
        workload=WorkloadSpec(
            num_requests=4,
            arrival_rate=1.0,
            prompt_tokens=32,
            output_tokens=4,
            seed=42,
        ),
    )
    result = engine.run(spec)
    assert len(result.request_results) == spec.workload.num_requests
    assert result.summary()["scheduler"] == "fcfs"
