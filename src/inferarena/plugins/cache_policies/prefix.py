"""Prefix cache that reuses exact prompt length prefixes across requests."""

from __future__ import annotations

from collections import OrderedDict

from inferarena.core.cache_policy import CachePolicy
from inferarena.core.request import Request


class PrefixCache(CachePolicy):
    """Exact-prefix cache with a bounded number of stored prefix lengths.

    Because the simulator does not expose raw token IDs, this implementation
    treats the prompt token count as the canonical prefix. Two requests with
    the same ``prompt_tokens`` value are considered to share a prefix.

    Attributes:
        max_prefixes: Maximum number of distinct prefix lengths to retain.
    """

    name = "prefix"

    def __init__(self, max_prefixes: int = 1024) -> None:
        """Initialize the prefix cache."""
        self.max_prefixes = max_prefixes
        self._cache: OrderedDict[int, None] = OrderedDict()

    def lookup(self, request: Request) -> int:
        """Return the longest cached prefix length for the request."""
        best = 0
        for prefix_length in self._cache:
            if prefix_length <= request.prompt_tokens:
                best = max(best, prefix_length)
        return best

    def store(self, request: Request) -> None:
        """Store the request's prompt length in the cache."""
        length = request.prompt_tokens
        if length in self._cache:
            self._cache.move_to_end(length)
            return
        self._cache[length] = None
        if len(self._cache) > self.max_prefixes:
            self._cache.popitem(last=False)
