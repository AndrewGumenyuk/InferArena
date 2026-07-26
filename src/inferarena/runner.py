"""High-level experiment runner for InferArena."""

from __future__ import annotations

from pathlib import Path

from inferarena.core.cache_policy import CachePolicy
from inferarena.core.execution_engine import ExecutionEngine
from inferarena.core.experiment_spec import ExperimentSpec
from inferarena.core.plugin_registry import PluginRegistry
from inferarena.core.scheduler import Scheduler
from inferarena.engines.sglang_adapter import SGLangEngine
from inferarena.engines.tensorrt_adapter import TensorRTEngine
from inferarena.engines.vidur_adapter import VidurEngine
from inferarena.engines.vllm_adapter import VLLMEngine
from inferarena.metrics.result import ExperimentResult
from inferarena.reporting.comparison import ComparisonReportGenerator
from inferarena.reporting.generator import ReportGenerator
from inferarena.simulation.engine import SimulationEngine
from inferarena.simulation.multi_gpu_engine import MultiGPUSimulationEngine


class ExperimentRunner:
    """Runs an experiment spec end-to-end."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        """Initialize the runner with a plugin registry."""
        self.registry = registry or PluginRegistry()

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run the experiment and return results."""
        scheduler_cls = self.registry.get_scheduler(spec.scheduler)
        scheduler = scheduler_cls()
        engine = self._create_engine(scheduler, spec)
        result = engine.run(spec)
        metrics = engine.metrics
        report = ReportGenerator.from_result(result, metrics)
        report.save(spec.output_dir)
        return result

    def compare(
        self, base_spec: ExperimentSpec, scheduler_names: list[str]
    ) -> list[ExperimentResult]:
        """Run the same workload with multiple schedulers and return results."""
        results: list[ExperimentResult] = []
        for name in scheduler_names:
            spec = base_spec.model_copy(deep=True)
            spec.scheduler = name
            spec.output_dir = base_spec.output_dir / name
            result = self.run(spec)
            results.append(result)
        ComparisonReportGenerator(results).save(base_spec.output_dir / "comparison")
        return results

    def benchmark(self, configs: list[ExperimentSpec]) -> list[ExperimentResult]:
        """Run a suite of benchmark configs and return results."""
        results: list[ExperimentResult] = []
        for spec in configs:
            result = self.run(spec)
            results.append(result)
        ComparisonReportGenerator(results).save(Path("inferarena_outputs/benchmarks/summary"))
        return results

    def _create_engine(self, scheduler: Scheduler, spec: ExperimentSpec) -> ExecutionEngine:
        """Create the execution engine based on the spec."""
        cache_policy = self._create_cache_policy(spec)
        if spec.engine.name == "simulation":
            return SimulationEngine(scheduler, spec.engine, cache_policy=cache_policy)
        if spec.engine.name == "multi_gpu_simulation":
            return MultiGPUSimulationEngine(scheduler, spec.engine, cache_policy=cache_policy)
        if spec.engine.name == "vllm":
            return VLLMEngine(scheduler, spec.engine, cache_policy=cache_policy)
        if spec.engine.name == "sglang":
            return SGLangEngine(scheduler, spec.engine, cache_policy=cache_policy)
        if spec.engine.name == "tensorrt":
            return TensorRTEngine(scheduler, spec.engine, cache_policy=cache_policy)
        if spec.engine.name == "vidur":
            return VidurEngine(scheduler, spec.engine)
        raise ValueError(f"Unsupported engine: {spec.engine.name}")

    def _create_cache_policy(self, spec: ExperimentSpec) -> CachePolicy:
        """Create the cache policy based on the spec."""
        cache_cls = self.registry.get_cache_policy(spec.cache_policy)
        return cache_cls()
