"""Worker abstraction for multi-GPU simulation."""

from __future__ import annotations

from inferarena.core.cache_policy import CachePolicy
from inferarena.core.experiment_spec import EngineSpec, WorkerSpec
from inferarena.core.request import Request
from inferarena.core.scheduler import Scheduler
from inferarena.metrics.result import ExperimentResult
from inferarena.simulation.engine import SimulationEngine


class Worker:
    """A single worker (GPU or GPU group) in a simulated cluster.

    Each worker owns a scheduler instance, a cache policy instance, and a
    ``SimulationEngine``. Requests routed to the worker are executed
    independently of other workers.
    """

    def __init__(
        self,
        worker_id: int,
        scheduler: Scheduler,
        engine_spec: EngineSpec,
        worker_spec: WorkerSpec,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a worker.

        Args:
            worker_id: Unique identifier for this worker.
            scheduler: Scheduler instance for this worker.
            engine_spec: Base engine specification.
            worker_spec: Worker-specific hardware/topology specification.
            cache_policy: Optional cache policy instance.
        """
        self.worker_id = worker_id
        self.worker_spec = worker_spec
        # Scale per-worker capacity by the number of GPUs it represents.
        effective_max_tokens = worker_spec.max_tokens_per_step * worker_spec.gpu_count
        self.engine = SimulationEngine(
            scheduler,
            engine_spec=engine_spec.model_copy(
                update={"max_tokens_per_step": effective_max_tokens}
            ),
            cache_policy=cache_policy,
        )

    def run(self, requests: list[Request], max_steps: int) -> ExperimentResult:
        """Run the assigned requests through the worker's engine."""
        return self.engine._run_requests(requests, max_steps)
