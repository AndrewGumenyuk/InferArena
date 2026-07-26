"""Multi-GPU discrete-event simulation engine."""

from __future__ import annotations

from inferarena.core.cache_policy import CachePolicy
from inferarena.core.execution_engine import ExecutionEngine
from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec
from inferarena.core.request import Request
from inferarena.core.scheduler import Scheduler
from inferarena.metrics.collector import MetricsCollector
from inferarena.metrics.result import ExperimentResult, RequestResult
from inferarena.simulation.router import LeastLoadedRouter, RoundRobinRouter, Router
from inferarena.simulation.worker import Worker


class MultiGPUSimulationEngine(ExecutionEngine):
    """Data-parallel multi-GPU simulator.

    Each worker owns a full model replica and runs an independent
    ``SimulationEngine``. A router assigns incoming requests to workers.
    Results are aggregated across workers.

    This engine models data-parallel serving. Tensor and pipeline parallelism
    are approximated by scaling each worker's token budget and adding a
    communication overhead constant to the engine spec.
    """

    name = "multi_gpu_simulation"

    def __init__(
        self,
        scheduler: Scheduler,
        engine_spec: EngineSpec | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize the multi-GPU simulation engine.

        Args:
            scheduler: Scheduler class or instance template for each worker.
            engine_spec: Base engine configuration.
            cache_policy: Optional cache policy for each worker.
        """
        self.scheduler = scheduler
        self.engine_spec = engine_spec or EngineSpec()
        self.cache_policy = cache_policy
        self.metrics = MetricsCollector()

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run the simulation across multiple workers."""
        from inferarena.workloads import load_workload

        requests = load_workload(spec.workload)
        workers = self._create_workers(spec)
        router = self._create_router(spec)

        # Route each request to a worker.
        assignments: dict[int, list[Request]] = {w.worker_id: [] for w in workers}
        for request in requests:
            worker = router.route(request, workers)
            assignments[worker.worker_id].append(request)

        # Run each worker independently.
        results = [worker.run(assignments[worker.worker_id], spec.max_steps) for worker in workers]

        return self._aggregate_results(results)

    def _create_workers(self, spec: ExperimentSpec) -> list[Worker]:
        """Create worker instances from the cluster spec."""
        cache_policy = self.cache_policy
        workers: list[Worker] = []
        for idx, worker_spec in enumerate(spec.cluster.workers):
            scheduler_instance = self.scheduler.__class__()
            worker_cache = cache_policy.__class__() if cache_policy else None
            workers.append(
                Worker(
                    worker_id=idx,
                    scheduler=scheduler_instance,
                    engine_spec=self.engine_spec,
                    worker_spec=worker_spec,
                    cache_policy=worker_cache,
                )
            )
        return workers

    def _create_router(self, spec: ExperimentSpec) -> Router:
        """Create the request router from the cluster spec."""
        name = spec.cluster.router
        if name == "round_robin":
            return RoundRobinRouter()
        if name == "least_loaded":
            return LeastLoadedRouter()
        raise ValueError(f"Unknown router: {name}")

    @staticmethod
    def _aggregate_results(results: list[ExperimentResult]) -> ExperimentResult:
        """Aggregate per-worker results into a single experiment result."""
        if not results:
            return ExperimentResult(
                scheduler_name="unknown",
                total_steps=0,
                total_time=0.0,
                completed_requests=0,
            )

        all_request_results: list[RequestResult] = []
        total_cache_hits = 0
        total_cache_lookups = 0
        for result in results:
            all_request_results.extend(result.request_results)
            total_cache_hits += result.cache_hits
            total_cache_lookups += result.cache_lookups

        return ExperimentResult(
            scheduler_name=results[0].scheduler_name,
            total_steps=max(r.total_steps for r in results),
            total_time=max(r.total_time for r in results),
            completed_requests=sum(r.completed_requests for r in results),
            request_results=all_request_results,
            cache_hits=total_cache_hits,
            cache_lookups=total_cache_lookups,
        )
