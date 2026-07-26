"""Tests for the round-robin scheduler."""

from inferarena.core.request import Request, RequestStatus
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.plugins.schedulers.round_robin import RoundRobinScheduler


def _state_with_waiting(
    requests: list[Request],
    max_tokens: int = 1000,
) -> SystemState:
    return SystemState(
        current_time=0.0,
        step=0,
        running=[],
        waiting=requests,
        completed=[],
        budget=TokenBudget(max_tokens=max_tokens),
    )


def test_round_robin_starts_from_cursor() -> None:
    scheduler = RoundRobinScheduler()
    requests = [
        Request(request_id="a", arrival_time=0.0, prompt_tokens=100, max_output_tokens=10),
        Request(request_id="b", arrival_time=1.0, prompt_tokens=100, max_output_tokens=10),
    ]
    for r in requests:
        r.status = RequestStatus.WAITING

    scheduler._cursor = 1
    batch = scheduler.schedule(_state_with_waiting(requests))
    assert batch.requests[0].request_id == "b"


def test_round_robin_rotates_starting_point() -> None:
    scheduler = RoundRobinScheduler()
    requests = [
        Request(request_id="a", arrival_time=0.0, prompt_tokens=100, max_output_tokens=10),
        Request(request_id="b", arrival_time=1.0, prompt_tokens=100, max_output_tokens=10),
        Request(request_id="c", arrival_time=2.0, prompt_tokens=100, max_output_tokens=10),
    ]
    for r in requests:
        r.status = RequestStatus.WAITING

    batch = scheduler.schedule(_state_with_waiting(requests))
    ids = [r.request_id for r in batch.requests]
    assert ids == ["a", "b", "c"]
    assert scheduler._cursor == 0

    # If budget only allows one, the next batch should start from b.
    scheduler2 = RoundRobinScheduler()
    for r in requests:
        r.status = RequestStatus.WAITING
    batch = scheduler2.schedule(_state_with_waiting(requests, max_tokens=100))
    assert batch.requests[0].request_id == "a"
    assert scheduler2._cursor == 1


def test_round_robin_respects_budget() -> None:
    scheduler = RoundRobinScheduler()
    requests = [
        Request(request_id="a", arrival_time=0.0, prompt_tokens=100, max_output_tokens=10),
        Request(request_id="b", arrival_time=1.0, prompt_tokens=100, max_output_tokens=10),
    ]
    for r in requests:
        r.status = RequestStatus.WAITING

    batch = scheduler.schedule(_state_with_waiting(requests, max_tokens=100))
    assert len(batch.requests) == 1
    assert batch.requests[0].request_id == "a"
