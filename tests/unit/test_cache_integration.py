"""Tests for cache policy integration with the simulation engine."""

from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec, WorkloadSpec
from inferarena.plugins.cache_policies.prefix import PrefixCache
from inferarena.plugins.schedulers.fcfs import FCFSScheduler
from inferarena.simulation.engine import SimulationEngine


def test_prefix_cache_reduces_prefill_time() -> None:
    scheduler = FCFSScheduler()
    cache_policy = PrefixCache(max_prefixes=10)
    engine = SimulationEngine(
        scheduler,
        EngineSpec(max_tokens_per_step=512, prefill_time_per_token=1.0),
        cache_policy=cache_policy,
    )

    # Two identical prompts: the second should hit the cache.
    spec = ExperimentSpec(
        workload=WorkloadSpec(
            num_requests=2,
            arrival_rate=1.0,
            prompt_tokens=10,
            output_tokens=2,
            seed=42,
        ),
    )
    result = engine.run(spec)
    assert result.completed_requests == 2
    assert result.cache_hits > 0
    assert result.cache_lookups > 0
    assert result.summary()["cache_hit_rate"] > 0.0
