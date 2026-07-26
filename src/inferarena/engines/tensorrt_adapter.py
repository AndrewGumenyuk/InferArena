"""Real-cluster execution engine backed by a TensorRT-LLM deployment.

TensorRT-LLM can be served through the Triton Inference Server with an
OpenAI-compatible front-end or via the ``trtllm-build`` Python runtime. This
engine connects to an OpenAI-compatible endpoint, replays a workload with
realistic inter-arrival timings, and collects per-request TTFT and end-to-end
latency.

Install the optional dependency::

    pip install -e ".[tensorrt]"

Start TensorRT-LLM separately; consult the TensorRT-LLM documentation for the
exact launch command for your model and engine build.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from inferarena.core.batch import Batch
from inferarena.core.cache_policy import CachePolicy
from inferarena.core.execution_engine import ExecutionEngine
from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec
from inferarena.core.request import Request
from inferarena.core.scheduler import Scheduler
from inferarena.metrics.collector import MetricsCollector
from inferarena.metrics.result import ExperimentResult, RequestResult

_LOGGER = logging.getLogger(__name__)


def _prompt_text(request: Request) -> str:
    """Build a dummy prompt with approximately the requested token count."""
    words = max(1, int(request.prompt_tokens * 0.75))
    return "hello " * words


class TensorRTEngine(ExecutionEngine):
    """Real-cluster execution engine backed by a TensorRT-LLM server."""

    name = "tensorrt"

    def __init__(
        self,
        scheduler: Scheduler,
        engine_spec: EngineSpec | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize the TensorRT-LLM adapter.

        Args:
            scheduler: InferArena scheduler (used for reporting, not runtime control).
            engine_spec: Engine configuration. Relevant fields:
                - ``model``: model name served by TensorRT-LLM.
                - ``base_url``: TensorRT-LLM API base URL.
                - ``api_key``: optional API key.
            cache_policy: Optional cache policy (not used at runtime).
        """
        self.scheduler = scheduler
        self.engine_spec = engine_spec or EngineSpec()
        self.cache_policy = cache_policy
        self.metrics = MetricsCollector()

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run the workload against the TensorRT-LLM deployment and return metrics."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "TensorRT-LLM engine requires the openai package. "
                "Install it with: pip install inferarena[tensorrt]"
            ) from exc

        from inferarena.workloads import load_workload

        requests = load_workload(spec.workload)
        client = AsyncOpenAI(
            base_url=self._base_url(),
            api_key=self._api_key(),
        )
        model = self._model()
        max_tokens = spec.workload.output_tokens

        results = asyncio.run(self._run_all(client, requests, model, max_tokens))

        completed = sum(1 for r in results if r.completion_time is not None)
        total_time = max(
            (r.completion_time or 0.0 for r in results),
            default=0.0,
        )
        self._record_metrics(results)
        return ExperimentResult(
            scheduler_name=self.scheduler.name,
            total_steps=len(results),
            total_time=total_time,
            completed_requests=completed,
            request_results=results,
        )

    async def _run_all(
        self,
        client: Any,
        requests: list[Request],
        model: str,
        max_tokens: int,
    ) -> list[RequestResult]:
        """Launch all requests according to their arrival times."""
        experiment_start = time.monotonic()
        results: list[RequestResult] = []
        tasks: list[asyncio.Task[RequestResult]] = []

        for request in requests:
            target = experiment_start + request.arrival_time / 1000.0
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            task = asyncio.create_task(
                self._call_api(client, request, model, max_tokens, experiment_start)
            )
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks)
        return results

    async def _call_api(
        self,
        client: Any,
        request: Request,
        model: str,
        max_tokens: int,
        experiment_start: float,
    ) -> RequestResult:
        """Call the TensorRT-LLM API for a single request and record timings."""
        result = RequestResult(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            prompt_tokens=request.prompt_tokens,
            max_output_tokens=max_tokens,
        )
        first_token_time: float | None = None
        start_ms = experiment_start * 1000.0

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt_text(request)}],
                max_tokens=max_tokens,
                stream=True,
            )
            async for _chunk in stream:
                now_ms = time.monotonic() * 1000.0 - start_ms
                if first_token_time is None:
                    first_token_time = now_ms
                    result.first_token_time = now_ms
            result.completion_time = time.monotonic() * 1000.0 - start_ms
        except Exception as exc:  # pragma: no cover - runtime errors are logged, not tested
            _LOGGER.warning("Request %s failed: %s", request.request_id, exc)

        return result

    def _record_metrics(self, results: list[RequestResult]) -> None:
        """Populate the metrics collector from per-request results."""
        self.metrics = MetricsCollector()
        completed = 0
        for step, result in enumerate(results, start=1):
            if result.completion_time is not None:
                completed += 1
            self.metrics.record_step(
                step=step,
                time=result.completion_time or result.arrival_time,
                batch=Batch(),
                waiting=0,
                running=len(results) - step,
                completed=completed,
                extra={
                    "first_token_time": result.first_token_time,
                    "completion_time": result.completion_time,
                },
            )

    def _base_url(self) -> str:
        """Return the TensorRT-LLM base URL from the engine spec or default."""
        return getattr(self.engine_spec, "base_url", "http://localhost:8000/v2")

    def _api_key(self) -> str:
        """Return the API key from the engine spec or a dummy default."""
        return getattr(self.engine_spec, "api_key", "dummy")

    def _model(self) -> str:
        """Return the model name from the engine spec or a default."""
        return getattr(self.engine_spec, "model", "ensemble")
