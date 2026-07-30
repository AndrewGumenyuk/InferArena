#!/usr/bin/env python3
"""Validate simulation against the mock OpenAI-compatible server.

Runs the same workload in simulation and against the mock engine, then
compares completion rates, latency distributions, and throughput.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec, WorkloadSpec
from inferarena.runner import ExperimentRunner


def build_spec(engine: EngineSpec, output_dir: str) -> ExperimentSpec:
    return ExperimentSpec(
        name="validation",
        scheduler="fcfs",
        cache_policy="no_op",
        workload=WorkloadSpec(
            name="variable",
            generator="variable",
            num_requests=8,
            arrival_rate=0.5,
            prompt_tokens_min=64,
            prompt_tokens_max=1024,
            output_tokens_min=16,
            output_tokens_max=64,
            seed=42,
        ),
        engine=engine,
        max_steps=20000,
        output_dir=Path(output_dir),
    )


def main() -> None:
    runner = ExperimentRunner()

    sim_spec = build_spec(
        EngineSpec(name="simulation", max_tokens_per_step=512),
        "./inferarena_outputs/validation/simulation",
    )
    real_spec = build_spec(
        EngineSpec(
            name="vllm",
            model="mock-model",
            base_url="http://localhost:8000/v1",
            api_key="dummy",
        ),
        "./inferarena_outputs/validation/mock_server",
    )

    print("Running simulation...")
    sim_result = runner.run(sim_spec)
    print("Running mock-server experiment...")
    real_result = runner.run(real_spec)

    comparison = {
        "simulation": sim_result.summary(),
        "mock_server": real_result.summary(),
    }

    output = Path("inferarena_outputs/validation/comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2))
    print(f"Saved {output}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
