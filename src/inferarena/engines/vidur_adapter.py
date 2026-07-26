"""Stub adapter for running InferArena experiments through Vidur.

Vidur is a high-fidelity LLM inference simulator from Georgia Tech / Microsoft
Research: https://arxiv.org/pdf/2405.05465

This module is intentionally a sketch. It will be implemented once Vidur is
available as a stable dependency and a concrete use case needs hardware-accurate
validation.
"""

from __future__ import annotations

from inferarena.core.execution_engine import ExecutionEngine
from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec
from inferarena.core.scheduler import Scheduler
from inferarena.metrics.result import ExperimentResult


class VidurEngine(ExecutionEngine):
    """High-fidelity execution engine backed by Vidur.

    Not yet functional; exists to document the intended integration point.
    """

    name = "vidur"

    def __init__(
        self,
        scheduler: Scheduler,
        engine_spec: EngineSpec | None = None,
    ) -> None:
        """Initialize the Vidur adapter.

        Args:
            scheduler: InferArena scheduler plugin to evaluate.
            engine_spec: Engine configuration including model and GPU specs.
        """
        self.scheduler = scheduler
        self.engine_spec = engine_spec or EngineSpec()

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run the experiment through Vidur.

        Raises:
            NotImplementedError: Vidur integration is not yet wired in.
        """
        raise NotImplementedError(
            "Vidur engine is a planned adapter. "
            "See docs/explanation/vidur-integration.md for the design."
        )
