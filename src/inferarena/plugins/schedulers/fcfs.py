"""First-come-first-served continuous batching scheduler."""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.request import RequestStatus
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class FCFSScheduler(Scheduler):
    """First-come-first-served scheduler with continuous batching.

    Prioritizes running decode requests, then admits waiting requests in arrival order.
    """

    name = "fcfs"

    def schedule(self, state: SystemState) -> Batch:
        """Select requests using FCFS continuous batching."""
        batch = Batch()
        used_tokens = 0

        # First, keep running decode requests in the batch.
        for request in state.running:
            if request.is_complete:
                continue
            cost = 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        # Then admit waiting requests in arrival order.
        sorted_waiting = sorted(state.waiting, key=lambda r: r.arrival_time)
        for request in sorted_waiting:
            if request.status != RequestStatus.WAITING:
                continue
            cost = request.prompt_tokens if not request.is_prefill_complete else 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        return batch
