"""Tests for the Sarathi-Serve scheduler reproduction."""

from inferarena.core.request import Request, RequestStatus
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.plugins.schedulers.sarathi_serve import SarathiServeScheduler


def test_sarathi_packs_decode_first() -> None:
    scheduler = SarathiServeScheduler(chunk_size=512)
    running = [
        Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=200, max_output_tokens=5),
    ]
    for r in running:
        r.status = RequestStatus.RUNNING
        r.prefilled_tokens = r.prompt_tokens
    waiting = [Request(arrival_time=2.0, prompt_tokens=300, max_output_tokens=5)]

    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=512),
    )
    batch = scheduler.schedule(state)

    assert len(batch.requests) == 3
    # Decode requests come first.
    assert batch.requests[0].is_prefill_complete
    assert batch.requests[1].is_prefill_complete
    # New request gets a chunk.
    assert not batch.requests[2].is_prefill_complete
    assert batch.tokens_for(batch.requests[2]) <= 512
    assert batch.total_tokens == 2 * 1 + 300


def test_sarathi_chunks_large_prefill() -> None:
    scheduler = SarathiServeScheduler(chunk_size=512)
    waiting = [Request(arrival_time=0.0, prompt_tokens=1024, max_output_tokens=5)]

    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=512),
    )
    batch = scheduler.schedule(state)

    assert len(batch.requests) == 1
    assert batch.tokens_for(waiting[0]) == 512
    assert batch.total_tokens == 512


def test_sarathi_respects_budget() -> None:
    scheduler = SarathiServeScheduler(chunk_size=512)
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=400, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=400, max_output_tokens=5),
    ]

    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=512),
    )
    batch = scheduler.schedule(state)

    # Only one request fits because each costs 400 tokens.
    assert len(batch.requests) == 1
    assert batch.total_tokens <= 512


def test_sarathi_continues_partial_prefill() -> None:
    scheduler = SarathiServeScheduler(chunk_size=512)
    running = [
        Request(arrival_time=0.0, prompt_tokens=1024, max_output_tokens=5),
    ]
    running[0].status = RequestStatus.RUNNING
    running[0].prefilled_tokens = 512

    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=[],
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=512),
    )
    batch = scheduler.schedule(state)

    assert len(batch.requests) == 1
    assert batch.tokens_for(running[0]) == 512
    assert batch.total_tokens == 512


def test_sarathi_batch_empty_when_budget_full() -> None:
    scheduler = SarathiServeScheduler(chunk_size=512)
    running = [
        Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5),
    ]
    running[0].status = RequestStatus.RUNNING
    running[0].prefilled_tokens = running[0].prompt_tokens

    waiting = [Request(arrival_time=1.0, prompt_tokens=1024, max_output_tokens=5)]

    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=1),
    )
    batch = scheduler.schedule(state)

    # Decode fills the budget; waiting request is not admitted.
    assert len(batch.requests) == 1
    assert batch.requests[0] is running[0]
