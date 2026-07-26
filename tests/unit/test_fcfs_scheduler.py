"""Tests for the FCFS scheduler."""

from inferarena.core.request import Request, RequestStatus
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.plugins.schedulers.fcfs import FCFSScheduler


def test_fcfs_admits_waiting_requests() -> None:
    scheduler = FCFSScheduler()
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=20, max_output_tokens=5),
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
    assert batch.requests[0].arrival_time <= batch.requests[1].arrival_time


def test_fcfs_respects_budget() -> None:
    scheduler = FCFSScheduler()
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=100, max_output_tokens=5),
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
    assert batch.total_tokens <= state.budget.max_tokens


def test_fcfs_keeps_running_requests() -> None:
    scheduler = FCFSScheduler()
    running = [
        Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5),
    ]
    running[0].status = RequestStatus.RUNNING
    running[0].scheduled_steps = 1
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=[],
        running=running,
        completed=[],
        budget=TokenBudget(max_tokens=10),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 1
    assert batch.requests[0] is running[0]
