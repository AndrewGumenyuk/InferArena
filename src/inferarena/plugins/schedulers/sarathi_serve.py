"""Faithful reproduction of the Sarathi-Serve scheduler.

Based on: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM
Inference with Sarathi-Serve" (arXiv:2403.02310).

Key ideas from the paper:
- Chunked prefills: split prefill into equal compute-sized chunks.
- Stall-free scheduling: new requests can join a running batch without
  pausing ongoing decodes.
- In each iteration:
  1. Pack all running decode requests.
  2. Include partially completed prefills.
  3. Admit new requests only after all running requests are accommodated.
  4. When adding prefill requests, cap the chunk at the remaining budget.
"""

from __future__ import annotations

from inferarena.core.batch import Batch
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class SarathiServeScheduler(Scheduler):
    """Sarathi-Serve scheduler with chunked prefill and stall-free batching.

    Attributes:
        chunk_size: Maximum number of prefill tokens to process in one step.
    """

    name = "sarathi_serve"

    def __init__(self, chunk_size: int = 512) -> None:
        """Initialize the scheduler with a prefill chunk size."""
        self.chunk_size = chunk_size

    def schedule(self, state: SystemState) -> Batch:
        """Build a batch following Sarathi-Serve's Algorithm 3."""
        batch = Batch()
        used_tokens = 0

        # 1. Pack all running decode requests first.
        for request in state.running:
            if request.is_complete or not request.is_prefill_complete:
                continue
            cost = 1
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            used_tokens += cost

        # 2. Include partially completed prefills (requests already running
        #    that still have prefill tokens left).
        for request in state.running:
            if request.is_complete or request.is_prefill_complete:
                continue
            cost = min(request.remaining_prefill, self.chunk_size)
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            batch.token_counts[request.request_id] = cost
            used_tokens += cost

        # 3. Admit new requests only after all running requests are packed.
        sorted_waiting = sorted(state.waiting, key=lambda r: r.arrival_time)
        for request in sorted_waiting:
            cost = min(request.remaining_prefill, self.chunk_size)
            if used_tokens + cost > state.budget.max_tokens:
                break
            batch.requests.append(request)
            batch.token_counts[request.request_id] = cost
            used_tokens += cost

        return batch
