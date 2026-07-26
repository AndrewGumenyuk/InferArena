"""Token budget abstraction for scheduling."""

from __future__ import annotations


class TokenBudget:
    """Budget limiting how many tokens can be processed in one step.

    Attributes:
        max_tokens: Maximum number of tokens allowed in the batch.
    """

    def __init__(self, max_tokens: int) -> None:
        """Initialize a token budget."""
        self.max_tokens = max_tokens

    def can_fit(self, tokens: int) -> bool:
        """Return True if the given token count fits within the budget."""
        return tokens <= self.max_tokens

    def __repr__(self) -> str:
        return f"TokenBudget(max_tokens={self.max_tokens})"
