"""No-op cache policy that never caches anything."""

from __future__ import annotations

from inferarena.core.cache_policy import CachePolicy
from inferarena.core.request import Request


class NoOpCachePolicy(CachePolicy):
    """Cache policy that always reports zero cached tokens."""

    name = "no_op"

    def lookup(self, request: Request) -> int:
        """Return zero cached tokens."""
        return 0

    def store(self, request: Request) -> None:
        """Do nothing."""
        return
