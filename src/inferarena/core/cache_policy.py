"""Base cache policy interface for inference experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferarena.core.request import Request


class CachePolicy(ABC):
    """Base class for cache policies that can reduce prefill work.

    A cache policy decides how much of a request's prompt is already cached
    from previous requests. Implementations may model prefix caching, page-based
    caching, or more sophisticated KV-cache reuse strategies.
    """

    name: str = "base"

    @abstractmethod
    def lookup(self, request: Request) -> int:
        """Return the number of cached prompt tokens for this request.

        Args:
            request: The incoming request.

        Returns:
            Number of prompt tokens that do not need to be recomputed.
        """
        ...

    @abstractmethod
    def store(self, request: Request) -> None:
        """Store the request's prompt in the cache after it has been processed.

        Args:
            request: The request that has completed prefill.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
