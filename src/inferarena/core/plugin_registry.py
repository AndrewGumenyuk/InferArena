"""Plugin registry for discovering and loading InferArena plugins."""

from __future__ import annotations

import importlib.metadata
from typing import TypeVar

from inferarena.core.cache_policy import CachePolicy
from inferarena.core.scheduler import Scheduler

T = TypeVar("T")


class PluginRegistry:
    """Registry for built-in and entry-point-discovered plugins.

    Plugins can be contributed by other packages via setuptools entry points:

    ```toml
    [project.entry-points."inferarena.schedulers"]
    my_scheduler = "my_package:MyScheduler"

    [project.entry-points."inferarena.cache_policies"]
    my_cache = "my_package:MyCachePolicy"
    ```
    """

    _SCHEDULER_GROUP = "inferarena.schedulers"
    _CACHE_POLICY_GROUP = "inferarena.cache_policies"

    def __init__(self) -> None:
        """Initialize the registry with built-in plugins."""
        self._schedulers: dict[str, type[Scheduler]] = {}
        self._cache_policies: dict[str, type[CachePolicy]] = {}
        self._load_built_ins()
        self._load_entry_points()

    def _load_built_ins(self) -> None:
        """Load built-in scheduler plugins."""
        from inferarena.plugins.cache_policies.no_op import NoOpCachePolicy
        from inferarena.plugins.cache_policies.prefix import PrefixCache
        from inferarena.plugins.schedulers.chunked_prefill import ChunkedPrefillScheduler
        from inferarena.plugins.schedulers.fcfs import FCFSScheduler
        from inferarena.plugins.schedulers.priority import PriorityScheduler
        from inferarena.plugins.schedulers.round_robin import RoundRobinScheduler
        from inferarena.plugins.schedulers.sjf import SJFScheduler

        self.register_scheduler(FCFSScheduler)
        self.register_scheduler(ChunkedPrefillScheduler)
        self.register_scheduler(PriorityScheduler)
        self.register_scheduler(SJFScheduler)
        self.register_scheduler(RoundRobinScheduler)

        self.register_cache_policy(NoOpCachePolicy)
        self.register_cache_policy(PrefixCache)

    def _load_entry_points(self) -> None:
        """Load scheduler plugins from entry points."""
        for ep in importlib.metadata.entry_points(group=self._SCHEDULER_GROUP):
            cls = ep.load()
            self.register_scheduler(cls)
        for ep in importlib.metadata.entry_points(group=self._CACHE_POLICY_GROUP):
            cls = ep.load()
            self.register_cache_policy(cls)

    def register_scheduler(self, cls: type[Scheduler]) -> None:
        """Register a scheduler class."""
        self._schedulers[cls.name] = cls

    def get_scheduler(self, name: str) -> type[Scheduler]:
        """Return a scheduler class by name."""
        if name not in self._schedulers:
            raise KeyError(f"Unknown scheduler: {name}")
        return self._schedulers[name]

    def list_schedulers(self) -> list[str]:
        """Return a sorted list of registered scheduler names."""
        return sorted(self._schedulers.keys())

    def register_cache_policy(self, cls: type[CachePolicy]) -> None:
        """Register a cache policy class."""
        self._cache_policies[cls.name] = cls

    def get_cache_policy(self, name: str) -> type[CachePolicy]:
        """Return a cache policy class by name."""
        if name not in self._cache_policies:
            raise KeyError(f"Unknown cache policy: {name}")
        return self._cache_policies[name]

    def list_cache_policies(self) -> list[str]:
        """Return a sorted list of registered cache policy names."""
        return sorted(self._cache_policies.keys())
