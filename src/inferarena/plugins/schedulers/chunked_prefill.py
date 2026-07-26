"""Chunked-prefill scheduler inspired by Sarathi-Serve."""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.request import RequestStatus
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class ChunkedPrefillScheduler(Scheduler):
    """Scheduler that breaks prefills into fixed-size chunks.

    This creates uniform-compute batches by mixing decode requests with small
    prefill chunks, avoiding generation stalls.

    Attributes:
        chunk_size: Maximum number of prompt tokens to process in one prefill step.
    """

    name = "chunked_prefill"

    def __init__(self, chunk_size: int = 512) -> None:
        """Initialize the scheduler with a chunk size."""
        self.chunk_size = chunk_size
        self._remaining_prefill: dict[str, int] = {}

    def schedule(self, state: SystemState) -> Batch:
        """Select requests using chunked prefills."""
        batch = Batch()
        used_tokens = 0

        # Keep running decode requests first (stall-free).
        for request in state.running:
            if request.is_complete:
                continue
            cost = 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        # Admit waiting requests, but cap prefill cost at chunk_size.
        sorted_waiting = sorted(state.waiting, key=lambda r: r.arrival_time)
        for request in sorted_waiting:
            if request.status != RequestStatus.WAITING:
                continue
            if request.request_id not in self._remaining_prefill:
                self._remaining_prefill[request.request_id] = request.prompt_tokens
            remaining = self._remaining_prefill[request.request_id]
            cost = min(remaining, self.chunk_size) if remaining > 0 else 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            self._remaining_prefill[request.request_id] = max(0, remaining - cost)
            used_tokens += cost

        # Clean up completed requests.
        completed_ids = {
            r.request_id for r in batch.requests if r.status == RequestStatus.COMPLETED
        }
        for request_id in completed_ids:
            self._remaining_prefill.pop(request_id, None)

        return batch
