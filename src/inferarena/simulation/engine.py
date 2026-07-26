"""Lightweight discrete-event simulation engine for scheduler evaluation."""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.cache_policy import CachePolicy
from inferarena.core.execution_engine import ExecutionEngine
from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec
from inferarena.core.request import Request, RequestStatus
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.metrics.collector import MetricsCollector
from inferarena.metrics.result import ExperimentResult, RequestResult
from inferarena.plugins.cache_policies.no_op import NoOpCachePolicy


class SimulationEngine(ExecutionEngine):
    """Discrete-event simulator for LLM inference scheduling policies."""

    name = "simulation"

    def __init__(
        self,
        scheduler: Scheduler,
        engine_spec: EngineSpec | None = None,
        metrics: MetricsCollector | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize the simulation engine.

        Args:
            scheduler: The scheduling policy to evaluate.
            engine_spec: Engine configuration.
            metrics: Optional metrics collector.
            cache_policy: Optional cache policy for prefix caching.
        """
        self.scheduler = scheduler
        self.engine_spec = engine_spec or EngineSpec()
        self.metrics = metrics or MetricsCollector()
        self.cache_policy = cache_policy or NoOpCachePolicy()
        self._time: float = 0.0
        self._step: int = 0
        self._waiting: list[Request] = []
        self._running: list[Request] = []
        self._completed: list[Request] = []
        self._request_results: dict[str, RequestResult] = {}

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run the simulation using the experiment spec.

        Args:
            spec: Experiment specification including workload and engine config.

        Returns:
            ExperimentResult with collected metrics.
        """
        from inferarena.workloads import load_workload

        requests = load_workload(spec.workload)
        return self._run_requests(requests, spec.max_steps)

    def _run_requests(self, requests: list[Request], max_steps: int) -> ExperimentResult:
        """Run the simulation on a list of requests."""
        pending = list(requests)
        self._time = 0.0
        self._step = 0
        self._waiting.clear()
        self._running.clear()
        self._completed.clear()
        self._request_results.clear()
        self.metrics = MetricsCollector()
        self._cache_hits = 0
        self._cache_lookups = 0

        while (pending or self._waiting or self._running) and self._step < max_steps:
            # Fast-forward to the next arrival when the system is idle.
            if pending and not self._waiting and not self._running:
                self._time = min(r.arrival_time for r in pending)

            # Admit newly arrived requests.
            newly_arrived = [r for r in pending if r.arrival_time <= self._time]
            for request in newly_arrived:
                request.status = RequestStatus.WAITING
                self._waiting.append(request)
                self._request_results[request.request_id] = RequestResult(
                    request_id=request.request_id,
                    arrival_time=request.arrival_time,
                    prompt_tokens=request.prompt_tokens,
                    max_output_tokens=request.max_output_tokens,
                )
            for request in newly_arrived:
                pending.remove(request)

            # Schedule the next batch.
            budget = TokenBudget(self.engine_spec.max_tokens_per_step)
            state = SystemState(
                current_time=self._time,
                step=self._step,
                waiting=list(self._waiting),
                running=list(self._running),
                completed=list(self._completed),
                budget=budget,
            )
            batch = self.scheduler.schedule(state)

            # Execute the batch.
            step_time = self._execute_batch(batch)
            self._time += step_time
            self._step += 1

            hit_rate = self._cache_hits / self._cache_lookups if self._cache_lookups else 0.0
            self.metrics.record_step(
                step=self._step,
                time=self._time,
                batch=batch,
                waiting=len(self._waiting),
                running=len(self._running),
                completed=len(self._completed),
                extra={
                    "cache_hits": self._cache_hits,
                    "cache_lookups": self._cache_lookups,
                    "cache_hit_rate": round(hit_rate, 4),
                },
            )

            # Move completed requests out of running.
            still_running: list[Request] = []
            for request in self._running:
                if request.is_complete:
                    request.status = RequestStatus.COMPLETED
                    self._completed.append(request)
                    result = self._request_results[request.request_id]
                    result.completion_time = self._time
                else:
                    still_running.append(request)
            self._running = still_running

        return ExperimentResult(
            scheduler_name=self.scheduler.name,
            total_steps=self._step,
            total_time=self._time,
            completed_requests=len(self._completed),
            request_results=list(self._request_results.values()),
            cache_hits=self._cache_hits,
            cache_lookups=self._cache_lookups,
        )

    def _execute_batch(self, batch: Batch) -> float:
        """Execute a batch and return the step duration in milliseconds."""
        step_time = 0.0
        cache_hits = 0
        cache_lookups = 0
        for request in batch.requests:
            if request.status == RequestStatus.WAITING:
                # First step: prefill.
                cached_tokens = self.cache_policy.lookup(request)
                cache_lookups += request.prompt_tokens
                cache_hits += min(cached_tokens, request.prompt_tokens)
                effective_prefill = max(1, request.prompt_tokens - cached_tokens)
                step_time = max(
                    step_time,
                    self.engine_spec.prefill_time_per_token * effective_prefill,
                )
                request.status = RequestStatus.RUNNING
                request.scheduled_steps += 1
                self._waiting.remove(request)
                self._running.append(request)
                result = self._request_results[request.request_id]
                result.scheduled_time = self._time
                result.first_token_time = self._time + step_time
                self.cache_policy.store(request)
            else:
                # Decode step.
                step_time = max(step_time, self.engine_spec.decode_time_per_token)
                request.advance(1)

        self._cache_hits += cache_hits
        self._cache_lookups += cache_lookups

        # Avoid zero time for empty steps.
        return max(step_time, 1.0)
