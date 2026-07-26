"""Base scheduler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferarena.core.batch import Batch
from inferarena.core.system_state import SystemState


class Scheduler(ABC):
    """Base class for all scheduling policies.

    A scheduler decides which waiting and running requests should be executed
    in the next engine step, subject to a token budget.
    """

    name: str = "base"

    @abstractmethod
    def schedule(self, state: SystemState) -> Batch:
        """Select requests to run in the next step.

        Args:
            state: Current system state including waiting/running requests and budget.

        Returns:
            A batch of requests to execute.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
