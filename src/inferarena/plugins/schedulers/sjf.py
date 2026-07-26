"""Shortest-job-first scheduler for low average latency."""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.request import RequestStatus
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class SJFScheduler(Scheduler):
    """Shortest-job-first scheduler with continuous batching.

    Prioritizes requests with the smallest total token budget to minimize
    average queueing and end-to-end latency.
    """

    name = "sjf"

    def schedule(self, state: SystemState) -> Batch:
        """Select requests using shortest-job-first ordering."""
        batch = Batch()
        used_tokens = 0

        # Keep running decode requests in the batch.
        for request in state.running:
            if request.is_complete:
                continue
            cost = 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        # Admit waiting requests by total work (prompt + max output).
        sorted_waiting = sorted(
            state.waiting,
            key=lambda r: (r.prompt_tokens + r.max_output_tokens, r.arrival_time),
        )
        for request in sorted_waiting:
            if request.status != RequestStatus.WAITING:
                continue
            cost = request.prompt_tokens if not request.is_prefill_complete else 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        return batch
