"""Base execution engine abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferarena.core.experiment_spec import ExperimentSpec
from inferarena.metrics.collector import MetricsCollector
from inferarena.metrics.result import ExperimentResult


class ExecutionEngine(ABC):
    """Abstract base class for experiment execution backends.

    Implementations include:
    - SimulationEngine: fast, GPU-free discrete-event simulation.
    - EmulationEngine: real engine stack with synthetic model latency.
    - RealClusterEngine: execution on actual inference servers.
    """

    name: str = "base"
    metrics: MetricsCollector

    @abstractmethod
    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run the experiment and return aggregated results."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
