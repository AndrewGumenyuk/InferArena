"""Tests for the shortest-job-first scheduler."""

from inferarena.core.request import Request, RequestStatus
from inferarena.core.system_state import SystemState
from inferarena.core.token_budget import TokenBudget
from inferarena.plugins.schedulers.sjf import SJFScheduler


def test_sjf_schedules_shortest_job_first() -> None:
    scheduler = SJFScheduler()
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=10, max_output_tokens=5),
    ]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=200),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 2
    # Shortest total work (10 + 5) should come before (100 + 5).
    assert batch.requests[0].prompt_tokens == 10
    assert batch.requests[1].prompt_tokens == 100


def test_sjf_keeps_running_decodes_first() -> None:
    scheduler = SJFScheduler()
    running = [Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5)]
    running[0].status = RequestStatus.RUNNING
    running[0].scheduled_steps = 1
    waiting = [Request(arrival_time=1.0, prompt_tokens=5, max_output_tokens=5)]
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
    assert batch.requests[0] is running[0]


def test_sjf_respects_budget() -> None:
    scheduler = SJFScheduler()
    waiting = [
        Request(arrival_time=0.0, prompt_tokens=50, max_output_tokens=5),
        Request(arrival_time=1.0, prompt_tokens=60, max_output_tokens=5),
    ]
    state = SystemState(
        current_time=0.0,
        step=0,
        waiting=waiting,
        running=[],
        completed=[],
        budget=TokenBudget(max_tokens=55),
    )
    batch = scheduler.schedule(state)
    assert len(batch.requests) == 1
    assert batch.requests[0].prompt_tokens == 50
