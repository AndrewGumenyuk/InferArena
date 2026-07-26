"""Priority-based scheduler for SLO-aware scheduling."""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.request import RequestStatus
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class PriorityScheduler(Scheduler):
    """Priority-based scheduler for SLO-aware inference.

    Higher-priority requests are scheduled first. Within the same priority,
    requests are ordered by arrival time.
    """

    name = "priority"

    def schedule(self, state: SystemState) -> Batch:
        """Select requests by priority."""
        batch = Batch()
        used_tokens = 0

        # Combine running and waiting; prefer keeping running decodes if high priority.
        all_requests = list(state.running) + list(state.waiting)
        sorted_requests = sorted(
            all_requests,
            key=lambda r: (-r.priority, r.arrival_time),
        )

        for request in sorted_requests:
            if request.is_complete:
                continue
            if request.status == RequestStatus.COMPLETED:
                continue
            cost = 1 if request.is_prefill_complete else request.prompt_tokens
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        return batch
