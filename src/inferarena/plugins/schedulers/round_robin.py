"""Round-robin scheduler for fair sharing across requests."""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.request import RequestStatus
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class RoundRobinScheduler(Scheduler):
    """Round-robin scheduler that rotates service across waiting requests.

    Unlike FCFS, which drains the oldest request first, round-robin gives each
    waiting request a small turn before moving to the next one. Running decode
    requests are still kept in the batch first to preserve continuous batching.
    """

    name = "round_robin"

    def __init__(self) -> None:
        """Initialize the scheduler with a rotating cursor."""
        self._cursor = 0

    def schedule(self, state: SystemState) -> Batch:
        """Select requests using round-robin continuous batching."""
        batch = Batch()
        used_tokens = 0

        # Keep running decode requests in the batch first.
        for request in state.running:
            if request.is_complete:
                continue
            cost = 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        # Sort waiting requests by arrival time so the cursor rotates over a
        # stable ordering.
        sorted_waiting = sorted(state.waiting, key=lambda r: r.arrival_time)
        if not sorted_waiting:
            return batch

        n = len(sorted_waiting)
        start_cursor = self._cursor
        admitted = 0
        for offset in range(n):
            idx = (start_cursor + offset) % n
            request = sorted_waiting[idx]
            if request.status != RequestStatus.WAITING:
                continue
            cost = request.prompt_tokens if not request.is_prefill_complete else 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost
            admitted += 1

        # Advance the cursor past the last examined request so the next step
        # starts from a rotated position.
        self._cursor = (start_cursor + admitted) % n

        return batch
