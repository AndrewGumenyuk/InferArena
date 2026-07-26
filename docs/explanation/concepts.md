# Core Concepts

InferArena is organized around a few simple ideas. Understanding them makes the rest of the documentation easier to follow.

## Inference component

An inference component is a pluggable decision strategy in the inference pipeline.

```text
InferenceComponent
       │
       ├── Scheduler
       ├── CachePolicy
       ├── Router
       └── (future: BatchPolicy, KVPolicy, MoEPolicy, ...)
```

You implement a component once and InferArena evaluates it across simulation and real inference engines.

## Scheduler

A scheduler decides which waiting and running requests execute in the next step, subject to a token budget.

Examples in InferArena:

- **FCFS** — first-come-first-served with continuous batching.
- **chunked_prefill** — splits large prefills into smaller chunks.
- **priority** — serves highest-priority requests first.
- **sjf** — shortest job first.
- **round_robin** — rotates service across requests.

## Cache policy

A cache policy decides how many prompt tokens are already cached, reducing the amount of prefill work.

- **no_op** — no caching.
- **prefix** — exact-prefix match cache.

## Router

A router assigns incoming requests to workers in multi-GPU simulations.

- **round_robin** — cycles across workers.
- **least_loaded** — picks the worker with the smallest queue.

## Engine

An engine executes the workload.

- **simulation** — fast, GPU-free discrete-event simulation.
- **multi_gpu_simulation** — data-parallel simulation across workers.
- **vllm**, **sglang**, **tensorrt** — real-cluster execution via OpenAI-compatible APIs.
- **vidur** — planned high-fidelity simulation adapter.

## Workload

A workload is the set of requests fed into an engine. Workloads can be:

- **uniform** — synthetic requests with fixed prompt/output lengths.
- **variable** — synthetic requests with randomized lengths.
- **trace** — requests loaded from a JSON/JSONL file such as ShareGPT.

## Experiment

An experiment ties together a scheduler, cache policy, workload, engine, and output directory. It is declared in a YAML config file and run with `inferarena run` or `inferarena compare`.

## Report

Every experiment produces a report directory with:

- `summary.json` — aggregated metrics.
- `requests.json` — per-request results.
- `telemetry.jsonl` — time-series events.
- `report.md` — human-readable summary.
- Optional plots (`latency_cdf.png`, `telemetry.png`).
