"""System state passed to scheduler plugins."""

from __future__ import annotations

from dataclasses import dataclass

from inferarena.core.request import Request
from inferarena.core.token_budget import TokenBudget


@dataclass
class SystemState:
    """Snapshot of the inference system at a scheduling decision point.

    Attributes:
        current_time: Current simulation/engine time in milliseconds.
        step: Current step number.
        waiting: Requests that have arrived but not started.
        running: Requests currently in flight.
        completed: Requests that have finished.
        budget: Token budget for the next batch.
    """

    current_time: float
    step: int
    waiting: list[Request]
    running: list[Request]
    completed: list[Request]
    budget: TokenBudget
