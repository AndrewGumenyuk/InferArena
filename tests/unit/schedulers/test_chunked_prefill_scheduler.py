"""Tests for the chunked-prefill scheduler."""

from inferarena.core.request import Request, RequestStatus
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.plugins.schedulers.chunked_prefill import ChunkedPrefillScheduler


def test_chunked_prefill_admits_first_chunk() -> None:
    scheduler = ChunkedPrefillScheduler(chunk_size=10)
    waiting = [Request(arrival_time=0.0, prompt_tokens=25, max_output_tokens=5)]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=10),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 1
    # The request fits only because the scheduler caps prefill cost at chunk_size.
    assert batch.requests[0] is waiting[0]


def test_chunked_prefill_rejects_when_budget_smaller_than_chunk() -> None:
    scheduler = ChunkedPrefillScheduler(chunk_size=10)
    waiting = [Request(arrival_time=0.0, prompt_tokens=25, max_output_tokens=5)]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=9),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 0


def test_chunked_prefill_tracks_remaining_prefill() -> None:
    scheduler = ChunkedPrefillScheduler(chunk_size=10)
    request = Request(arrival_time=0.0, prompt_tokens=25, max_output_tokens=5)
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=[request],
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=100),
    )

    # First scheduling consumes the first chunk.
    scheduler.schedule(state)
    assert scheduler._remaining_prefill[request.request_id] == 15

    # Simulate the engine marking the request as running.
    request.status = RequestStatus.RUNNING
    request.scheduled_steps = 1
    running_state = SystemState(
        current_time=0.0,
        step=1,
        waiting=[],
        running=[request],
        completed=[],
        budget=TokenBudget(max_tokens=100),
    )

    # Running requests are treated as decode with cost 1.
    batch = scheduler.schedule(running_state)
    assert len(batch.requests) == 1


def test_chunked_prefill_keeps_running_decodes_first() -> None:
    scheduler = ChunkedPrefillScheduler(chunk_size=512)
    running = [Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5)]
    running[0].status = RequestStatus.RUNNING
    running[0].scheduled_steps = 1
    waiting = [Request(arrival_time=1.0, prompt_tokens=512, max_output_tokens=5)]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=513),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 2
    assert batch.requests[0] is running[0]


def test_chunked_prefill_respects_budget() -> None:
    scheduler = ChunkedPrefillScheduler(chunk_size=100)
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=200, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=200, max_output_tokens=5),
    ]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=150),
    )
    batch = scheduler.schedule(state)
    # Only the first chunk of the first request fits within the budget.
    assert len(batch.requests) == 1
    assert batch.requests[0] is waiting[0]
