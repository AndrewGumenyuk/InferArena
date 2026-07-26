"""Tests for the plugin registry."""

from inferarena.core.plugin_registry import PluginRegistry
from inferarena.plugins.schedulers.fcfs import FCFSScheduler


def test_registry_loads_built_in_schedulers() -> None:
    registry = PluginRegistry()
    names = registry.list_schedulers()
    assert "fcfs" in names
    assert "chunked_prefill" in names
    assert "priority" in names


def test_registry_returns_scheduler_class() -> None:
    registry = PluginRegistry()
    cls = registry.get_scheduler("fcfs")
    assert cls is FCFSScheduler
