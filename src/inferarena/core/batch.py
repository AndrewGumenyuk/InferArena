"""Batch model for scheduled requests."""

from __future__ import annotations

from inferarena.core.request import Request


class Batch:
    """A batch of requests selected to run in the next step.

    Attributes:
        requests: List of requests in the batch.
        token_counts: Optional mapping from request_id to the number of tokens
            to process in this step. If omitted, defaults to full prefill for
            prefill-incomplete requests and 1 token for decode requests.
    """

    def __init__(
        self,
        requests: list[Request] | None = None,
        token_counts: dict[str, int] | None = None,
    ) -> None:
        """Initialize a batch."""
        self.requests: list[Request] = requests or []
        self.token_counts: dict[str, int] = token_counts or {}

    def tokens_for(self, request: Request) -> int:
        """Return the number of tokens to process for the given request."""
        if request.request_id in self.token_counts:
            return self.token_counts[request.request_id]
        if request.is_prefill_complete:
            return 1
        return request.remaining_prefill

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens scheduled in this batch."""
        return sum(self.tokens_for(r) for r in self.requests)

    def __len__(self) -> int:
        return len(self.requests)

    def __repr__(self) -> str:
        return f"Batch({len(self.requests)} requests, {self.total_tokens} tokens)"
