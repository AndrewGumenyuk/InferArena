"""Declarative experiment specification."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class WorkloadSpec(BaseModel):
    """Specification for a workload."""

    name: str = "uniform"
    trace_path: Path | None = None
    trace_format: str = "sharegpt"
    num_requests: int = 32
    arrival_rate: float = 2.0
    prompt_tokens: int = 512
    output_tokens: int = 128
    prompt_tokens_min: int = 256
    prompt_tokens_max: int = 512
    output_tokens_min: int = 64
    output_tokens_max: int = 128
    seed: int = 42


class EngineSpec(BaseModel):
    """Specification for an execution engine."""

    name: str = "simulation"
    max_tokens_per_step: int = 2048
    prefill_time_per_token: float = 0.1
    decode_time_per_token: float = 20.0


class WorkerSpec(BaseModel):
    """Specification for a single worker in a multi-GPU cluster."""

    gpu_count: int = 1
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    max_tokens_per_step: int = 2048
    inter_gpu_bandwidth_gbps: float = 400.0


class ClusterSpec(BaseModel):
    """Specification for a multi-GPU inference cluster."""

    workers: list[WorkerSpec] = Field(default_factory=lambda: [WorkerSpec()])
    router: str = "round_robin"


class ExperimentSpec(BaseModel):
    """Complete specification for a InferArena experiment."""

    name: str = "experiment"
    scheduler: str = "fcfs"
    cache_policy: str = "no_op"
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    engine: EngineSpec = Field(default_factory=EngineSpec)
    cluster: ClusterSpec = Field(default_factory=ClusterSpec)
    max_steps: int = 10000
    output_dir: Path = Path("./inferarena_outputs")

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentSpec:
        """Load an experiment spec from a YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
