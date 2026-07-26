"""Request model for inference experiments."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum


class RequestStatus(str, Enum):
    """Status of a request during an experiment."""

    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    PREEMPTED = "preempted"


class Request:
    """A single inference request in an experiment.

    Attributes:
        request_id: Unique identifier for the request.
        arrival_time: Timestamp when the request arrived (seconds from experiment start).
        prompt_tokens: Number of tokens in the prompt.
        max_output_tokens: Maximum number of output tokens to generate.
        status: Current status of the request.
        scheduled_steps: Number of steps the request has been scheduled.
        generated_tokens: Number of tokens generated so far.
        priority: Optional priority value (higher = more important).
    """

    def __init__(  # noqa: PLR0913
        self,
        arrival_time: float,
        prompt_tokens: int,
        max_output_tokens: int,
        priority: int = 0,
        request_id: str | None = None,
    ) -> None:
        """Initialize a request."""
        self.request_id = request_id or str(uuid.uuid4())
        self.arrival_time = arrival_time
        self.prompt_tokens = prompt_tokens
        self.max_output_tokens = max_output_tokens
        self.priority = priority
        self.status = RequestStatus.WAITING
        self.scheduled_steps = 0
        self.generated_tokens = 0
        self._created_at = datetime.now().isoformat()

    @property
    def is_prefill_complete(self) -> bool:
        """Return True if the prefill phase is complete."""
        return self.scheduled_steps > 0

    @property
    def is_complete(self) -> bool:
        """Return True if generation is complete."""
        return self.generated_tokens >= self.max_output_tokens

    def advance(self, tokens: int = 1) -> None:
        """Advance the request by the given number of tokens."""
        self.generated_tokens += tokens
        self.scheduled_steps += 1
        if self.is_complete:
            self.status = RequestStatus.COMPLETED

    def __repr__(self) -> str:
        return (
            f"Request({self.request_id}, "
            f"arrival={self.arrival_time:.3f}, "
            f"prompt={self.prompt_tokens}, "
            f"max_output={self.max_output_tokens})"
        )
