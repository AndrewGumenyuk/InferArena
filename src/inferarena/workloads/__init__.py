"""Workload generation and loading entry points."""

from __future__ import annotations

from inferarena.core.experiment_spec import WorkloadSpec
from inferarena.core.request import Request
from inferarena.workloads.synthetic import generate_uniform, generate_variable
from inferarena.workloads.trace import TraceWorkloadLoader


def load_workload(spec: WorkloadSpec) -> list[Request]:
    """Load or generate a workload from the given spec.

    Args:
        spec: Workload specification.

    Returns:
        A list of Request objects sorted by arrival time.

    Raises:
        ValueError: If the workload name is unknown or the spec is invalid.
    """
    if spec.trace_path is not None:
        return TraceWorkloadLoader(spec).load()

    if spec.name == "uniform":
        return generate_uniform(
            num_requests=spec.num_requests,
            arrival_rate=spec.arrival_rate,
            prompt_tokens=spec.prompt_tokens,
            max_output_tokens=spec.output_tokens,
            seed=spec.seed,
        )

    if spec.name == "variable":
        return generate_variable(
            num_requests=spec.num_requests,
            arrival_rate=spec.arrival_rate,
            prompt_tokens_range=(spec.prompt_tokens_min, spec.prompt_tokens_max),
            output_tokens_range=(spec.output_tokens_min, spec.output_tokens_max),
            seed=spec.seed,
        )

    raise ValueError(f"Unknown workload: {spec.name}")
