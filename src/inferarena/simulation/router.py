"""Request routers for multi-GPU simulation."""

from __future__ import annotations

from inferarena.core.request import Request
from inferarena.simulation.worker import Worker


class Router:
    """Base class for request routers."""

    name: str = "base"

    def route(self, request: Request, workers: list[Worker]) -> Worker:
        """Select a worker for the given request."""
        raise NotImplementedError


class RoundRobinRouter(Router):
    """Distributes requests evenly across workers in cyclic order."""

    name = "round_robin"

    def __init__(self) -> None:
        """Initialize the round-robin router."""
        self._next_index = 0

    def route(self, request: Request, workers: list[Worker]) -> Worker:
        """Select the next worker in the cycle."""
        worker = workers[self._next_index % len(workers)]
        self._next_index += 1
        return worker


class LeastLoadedRouter(Router):
    """Routes each request to the worker with the fewest assigned requests."""

    name = "least_loaded"

    def route(self, request: Request, workers: list[Worker]) -> Worker:
        """Select the worker with the smallest pending queue."""
        return min(workers, key=lambda w: len(w.engine._waiting) + len(w.engine._running))
