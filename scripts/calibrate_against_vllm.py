"""Calibrate the InferArena simulator against a real vLLM deployment.

Three phases:

A. Microbenchmark -- measure prefill_time_per_token (TTFT vs prompt size,
   using exact token counts from the API's usage field) and
   decode_time_per_token (median inter-token gap while streaming).
B. Replay -- run a fixed workload against the live vLLM server (API replay)
   and through the simulator with the calibrated parameters.
C. Compare -- write comparison.json and calibration_report.md.

Usage (server must already be running, e.g. `vllm serve Qwen/Qwen2.5-0.5B
--max-num-batched-tokens 2048`)::

    python scripts/calibrate_against_vllm.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-0.5B \
        --max-tokens-per-step 2048

Absolute parity is NOT the goal: the simulator has no network, tokenizer, or
sampling overhead (see docs/explanation/simulation-assumptions.md). Success is
matching trends and order of magnitude.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from inferarena.core.experiment_spec import (
    EngineSpec,
    ExperimentSpec,
    WorkloadSpec,
)
from inferarena.engines.vllm_adapter import VLLMEngine
from inferarena.plugins.schedulers.chunked_prefill import ChunkedPrefillScheduler
from inferarena.plugins.schedulers.fcfs import FCFSScheduler
from inferarena.simulation.engine import SimulationEngine

PREFILL_SIZES = [64, 128, 256, 512, 1024]
DECODE_BENCH_TOKENS = 64


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit y = slope*x + intercept (no numpy dependency)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den else 0.0
    return slope, mean_y - slope * mean_x


async def _measure_prefill(client: object, model: str) -> tuple[float, float, list[dict]]:
    """TTFT vs prompt size with max_tokens=1; returns (slope, intercept, points)."""
    points: list[dict] = []
    for target_tokens in PREFILL_SIZES:
        # ~0.75 words per token; usage.prompt_tokens gives the exact count.
        prompt = "hello " * int(target_tokens * 0.75)
        start = time.perf_counter()
        response = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        actual_tokens = response.usage.prompt_tokens if response.usage else target_tokens
        points.append(
            {"target": target_tokens, "actual_tokens": actual_tokens, "ttft_ms": elapsed_ms}
        )
    xs = [float(p["actual_tokens"]) for p in points]
    ys = [float(p["ttft_ms"]) for p in points]
    slope, intercept = _linear_fit(xs, ys)
    return slope, intercept, points


async def _measure_decode(client: object, model: str) -> tuple[float, list[float]]:
    """Median inter-token gap while streaming; concurrency 1."""
    chunk_times: list[float] = []
    stream = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 100, separated by spaces."}],
        max_tokens=DECODE_BENCH_TOKENS,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunk_times.append(time.perf_counter())
    # Gaps between consecutive content chunks; the first chunk's delay is prefill.
    gaps = [(b - a) * 1000.0 for a, b in zip(chunk_times, chunk_times[1:], strict=True)]
    median_tbt = statistics.median(gaps) if gaps else 0.0
    return median_tbt, gaps


def _run_simulation(
    spec: ExperimentSpec,
    prefill_ms: float,
    decode_ms: float,
    max_tokens_per_step: int,
) -> dict:
    """Run the same workload through the calibrated simulator."""
    engine_spec = EngineSpec(
        max_tokens_per_step=max_tokens_per_step,
        prefill_time_per_token=prefill_ms,
        decode_time_per_token=decode_ms,
    )
    engine = SimulationEngine(ChunkedPrefillScheduler(), engine_spec=engine_spec)
    return engine.run(spec).summary()


def _run_replay(spec: ExperimentSpec, base_url: str, api_key: str, model: str) -> dict:
    """Run the same workload against the live vLLM server."""
    engine_spec = EngineSpec(model=model, base_url=base_url, api_key=api_key)
    engine = VLLMEngine(FCFSScheduler(), engine_spec=engine_spec)
    return engine.run(spec).summary()


async def _main(args: argparse.Namespace) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit('The openai package is required: pip install -e ".[vllm]"') from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    print("Phase A: microbenchmark")
    prefill_ms, prefill_intercept, points = await _measure_prefill(client, args.model)
    decode_ms, gaps = await _measure_decode(client, args.model)
    print(
        f"  prefill_time_per_token = {prefill_ms:.4f} ms/token "
        f"(fixed overhead {prefill_intercept:.1f} ms)"
    )
    print(
        f"  decode_time_per_token  = {decode_ms:.2f} ms/token "
        f"(median of {len(gaps)} gaps, concurrency 1)"
    )

    print("Phase B: workload replay (real vs calibrated simulation)")
    spec = ExperimentSpec(
        name="calibration",
        workload=WorkloadSpec(
            name="variable",
            num_requests=args.requests,
            arrival_rate=args.arrival_rate,
            seed=args.seed,
        ),
        max_steps=args.max_steps,
    )
    real = _run_replay(spec, args.base_url, args.api_key, args.model)
    print(
        f"  real vLLM:    {real['completed_requests']}/{args.requests} completed, "
        f"TTFT p50 {real['ttft_p50_ms']} ms"
    )
    sim = _run_simulation(spec, prefill_ms, decode_ms, args.max_tokens_per_step)
    print(
        f"  calibrated:   {sim['completed_requests']}/{args.requests} completed, "
        f"TTFT p50 {sim['ttft_p50_ms']} ms"
    )

    print("Phase C: comparison")
    comparison = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "requests": args.requests,
            "arrival_rate": args.arrival_rate,
            "seed": args.seed,
            "max_tokens_per_step": args.max_tokens_per_step,
        },
        "calibration": {
            "prefill_time_per_token_ms": round(prefill_ms, 5),
            "prefill_fixed_overhead_ms": round(prefill_intercept, 2),
            "decode_time_per_token_ms": round(decode_ms, 3),
            "prefill_points": points,
        },
        "real_vllm": real,
        "calibrated_simulation": sim,
        "ratio_sim_over_real": {
            key: round(sim[key] / real[key], 3) if real[key] else None
            for key in ("throughput_rps", "ttft_p50_ms", "ttft_p99_ms", "latency_p50_ms")
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    report = f"""# Calibration report

Model: `{args.model}` — {args.requests} requests at {args.arrival_rate} req/s (seed {args.seed})

## Measured per-token costs

| Parameter | Value |
|---|---|
| prefill_time_per_token | {prefill_ms:.4f} ms |
| prefill fixed overhead | {prefill_intercept:.1f} ms |
| decode_time_per_token | {decode_ms:.2f} ms (concurrency 1) |

## Real vLLM vs calibrated simulation

| Metric | Real vLLM | Simulation | sim/real |
|---|---|---|---|
| Completed | {real["completed_requests"]} | {sim["completed_requests"]} | — |
| Throughput (rps) | {real["throughput_rps"]} | {sim["throughput_rps"]} | {comparison["ratio_sim_over_real"]["throughput_rps"]} |
| TTFT p50 (ms) | {real["ttft_p50_ms"]} | {sim["ttft_p50_ms"]} | {comparison["ratio_sim_over_real"]["ttft_p50_ms"]} |
| TTFT p99 (ms) | {real["ttft_p99_ms"]} | {sim["ttft_p99_ms"]} | {comparison["ratio_sim_over_real"]["ttft_p99_ms"]} |
| Latency p50 (ms) | {real["latency_p50_ms"]} | {sim["latency_p50_ms"]} | {comparison["ratio_sim_over_real"]["latency_p50_ms"]} |

## Caveats

- The simulator models no network, tokenizer, or sampling overhead, so real
  TTFT includes a fixed cost the simulation lacks (see prefill intercept).
- Decode time in the simulator is flat; on real hardware it grows with batch
  size. The measured value is at concurrency 1.
- Success criterion: matching trends and order of magnitude, not parity.
"""
    (out_dir / "calibration_report.md").write_text(report)
    print(f"Wrote {out_dir / 'comparison.json'} and {out_dir / 'calibration_report.md'}")


def main() -> None:
    """Parse arguments and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", required=True, help="Model name served by vLLM")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument(
        "--max-tokens-per-step",
        type=int,
        default=2048,
        help="Must match the server's --max-num-batched-tokens",
    )
    parser.add_argument("--output-dir", default="inferarena_outputs/calibration")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
