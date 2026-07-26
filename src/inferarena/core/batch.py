"""Batch model for scheduled requests."""

from __future__ import annotations

from inferarena.core.request import Request


class Batch:
    """A batch of requests selected to run in the next step.

    Attributes:
        requests: List of requests in the batch.
        prefill_requests: Subset of requests that are still in prefill.
        decode_requests: Subset of requests that are in decode.
    """

    def __init__(self, requests: list[Request] | None = None) -> None:
        """Initialize a batch."""
        self.requests: list[Request] = requests or []

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens scheduled in this batch."""
        total = 0
        for request in self.requests:
            if request.is_prefill_complete:
                total += 1  # decode step: one new token per request
            else:
                total += request.prompt_tokens
        return total

    def __len__(self) -> int:
        return len(self.requests)

    def __repr__(self) -> str:
        return f"Batch({len(self.requests)} requests, {self.total_tokens} tokens)"
