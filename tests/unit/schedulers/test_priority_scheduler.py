"""Tests for the priority scheduler."""

from inferarena.core.request import Request, RequestStatus
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.plugins.schedulers.priority import PriorityScheduler


def test_priority_schedules_high_priority_first() -> None:
    scheduler = PriorityScheduler()
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5, priority=1),
        Request(arrival_time=1.0, prompt_tokens=10, max_output_tokens=5, priority=5),
    ]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=100),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 2
    assert batch.requests[0].priority == 5
    assert batch.requests[1].priority == 1


def test_priority_tie_breaks_by_arrival_time() -> None:
    scheduler = PriorityScheduler()
    waiting = [
        Request(arrival_time=2.0, prompt_tokens=10, max_output_tokens=5, priority=1),
        Request(arrival_time=0.5, prompt_tokens=10, max_output_tokens=5, priority=1),
        Request(arrival_time=1.0, prompt_tokens=10, max_output_tokens=5, priority=1),
    ]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=100),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 3
    assert batch.requests[0].arrival_time == 0.5
    assert batch.requests[1].arrival_time == 1.0
    assert batch.requests[2].arrival_time == 2.0


def test_priority_respects_budget() -> None:
    scheduler = PriorityScheduler()
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5, priority=10),
        Request(arrival_time=1.0, prompt_tokens=100, max_output_tokens=5, priority=5),
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
    assert len(batch.requests) == 1
    assert batch.requests[0].priority == 10
    assert batch.total_tokens <= state.budget.max_tokens


def test_priority_prefers_high_priority_over_running_decode() -> None:
    scheduler = PriorityScheduler()
    running = [
        Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5, priority=0),
    ]
    running[0].status = RequestStatus.RUNNING
    running[0].scheduled_steps = 1
    waiting = [
        Request(arrival_time=1.0, prompt_tokens=10, max_output_tokens=5, priority=10),
    ]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=100),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 2
    # High-priority waiting request is scheduled before the low-priority running decode.
    assert batch.requests[0].priority == 10
    assert batch.requests[1].priority == 0


def test_priority_running_decode_cost_is_one() -> None:
    scheduler = PriorityScheduler()
    running = [
        Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5, priority=10),
    ]
    running[0].status = RequestStatus.RUNNING
    running[0].scheduled_steps = 1
    running[0].prefilled_tokens = running[0].prompt_tokens
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=[],
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=1),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 1
    assert batch.total_tokens == 1
