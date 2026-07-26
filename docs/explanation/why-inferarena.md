# Why InferArena?

A design doc for anyone asking: *"Why would a vLLM contributor use InferArena instead of writing a benchmark script?"*

## The problem

Improving LLM inference is one of the highest-leverage problems in AI infrastructure. A 10% reduction in serving cost at scale can save millions of dollars per year. Yet the research loop for new scheduling, caching, or routing ideas is surprisingly slow and brittle.

Today, if you have an idea for a better scheduler, you typically do the following:

1. Fork an inference engine such as vLLM, SGLang, or TensorRT-LLM.
2. Learn its internal scheduler, data structures, and build system.
3. Modify the scheduler and hope your change does not break other features.
4. Write ad-hoc scripts to generate a workload, run the experiment, and collect metrics.
5. Re-implement or copy FCFS and other baselines so you have something to compare against.
6. Manually produce plots and tables for a paper or PR.
7. Discover later that your benchmark is hard to reproduce because it depends on your local setup.

Each step adds friction. The result is that many good ideas are never evaluated fairly, and many evaluations are never reproduced.

## The alternative

InferArena treats the inference strategy as a plugin, separate from the engine that executes it.

You implement your idea once, as a small class that conforms to a stable interface. InferArena handles the rest: workload generation, execution across simulation or real engines, metric collection, baseline comparison, and report generation.

```
Idea
  ↓
class MyScheduler(Scheduler)
  ↓
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,priority,my_scheduler
  ↓
Reproducible report
```

## What a vLLM contributor gets

### A stable research interface

Instead of editing vLLM's internal scheduler, you write:

```python
class MyScheduler(Scheduler):
    name = "my_scheduler"

    def schedule(self, state: SystemState) -> Batch: ...
```

The interface is intentionally small. It exposes the system state and asks for a batch. The engine is responsible for everything else.

### Fair baselines for free

InferArena ships with FCFS, chunked prefill, priority, shortest-job-first, and round-robin schedulers, plus no-op and prefix cache policies. You do not have to re-implement baselines or argue about whether your comparison is apples-to-apples.

### Two execution modes

- **Simulation**: iterate in seconds without GPUs.
- **Real engines**: validate on vLLM, SGLang, or TensorRT-LLM through OpenAI-compatible APIs.

The same scheduler runs in both modes. If your idea works in simulation and holds up on a real engine, you have strong evidence. If it does not, you know where to dig.

### Standard, reproducible reports

Every run produces:

- `summary.json` with TTFT, latency, throughput, queue time, and cache metrics.
- `telemetry.jsonl` for time-series analysis.
- `requests.json` for per-request inspection.
- Optional latency CDF and telemetry plots.
- A comparison report when multiple schedulers are run.

Because outputs are plain files, they can be version-controlled, shared, or fed into the Streamlit dashboard.

### A path back to the engine

InferArena is not a replacement for vLLM. It is a research accelerator. When a scheduler proves valuable in InferArena, the next step is to port the policy into the production engine with confidence that it beats standard baselines.

## What InferArena does not do

- It does not implement production features such as fault tolerance, model sharding, or production observability.
- It does not claim to match the exact timing of a real GPU cluster in simulation.
- It does not force you to use one inference engine.

It is a shared substrate for experimenting with inference strategies before committing to a full engine integration.

## Design principles

1. **Research over execution.** The scheduler plugin is the star. The engine is interchangeable.
2. **Fair comparison.** Built-in baselines and the same workload for every strategy.
3. **Reproducibility.** Plain-file outputs, version-controlled configs, and deterministic workloads.
4. **Incremental growth.** Start with simulation. Add real engines, custom workloads, and new component types only when you need them.

## When to use InferArena

Use it when you want to answer questions like:

- *"Would a shortest-job-first scheduler reduce tail latency on my workload?"*
- *"How much does prefix caching help on ShareGPT-style prompts?"*
- *"Does my new batching idea beat FCFS across synthetic and real traces?"*

Do not use it when you are debugging a production outage or tuning a specific deployment. Use the production engine's tooling for that.

## The bigger vision

Today InferArena focuses on schedulers, cache policies, and routers. The long-term goal is a unified platform for *inference components*: any pluggable decision in the inference pipeline, including batching, prefix caching, KV-cache eviction, MoE routing, and strategies not yet invented.

The interface stays small. The platform grows around it.
