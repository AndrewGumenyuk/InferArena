"""Per-step metrics collector for InferArena experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferarena.core.batch import Batch


@dataclass
class StepMetrics:
    """Metrics captured for a single experiment step."""

    step: int
    time: float
    batch_size: int
    batch_tokens: int
    waiting_count: int
    running_count: int
    completed_count: int
    extra: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects telemetry across experiment steps."""

    def __init__(self) -> None:
        """Initialize the collector."""
        self._steps: list[StepMetrics] = []

    def record_step(
        self,
        step: int,
        time: float,
        batch: Batch,
        waiting: int,
        running: int,
        completed: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record metrics for one step."""
        self._steps.append(
            StepMetrics(
                step=step,
                time=time,
                batch_size=len(batch),
                batch_tokens=batch.total_tokens,
                waiting_count=waiting,
                running_count=running,
                completed_count=completed,
                extra=extra or {},
            )
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return collected metrics as a list of dictionaries."""
        return [
            {
                "step": m.step,
                "time": m.time,
                "batch_size": m.batch_size,
                "batch_tokens": m.batch_tokens,
                "waiting_count": m.waiting_count,
                "running_count": m.running_count,
                "completed_count": m.completed_count,
                **m.extra,
            }
            for m in self._steps
        ]

    @property
    def step_count(self) -> int:
        """Return the number of recorded steps."""
        return len(self._steps)
