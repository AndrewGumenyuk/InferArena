# InferArena

[![CI](https://github.com/AndrewGumenyuk/InferArena/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewGumenyuk/InferArena/actions/workflows/ci.yml)

> The open-source experimentation platform for LLM inference systems.

Implement an inference strategy once. Evaluate it consistently across simulation and production inference engines. Compare against standard baselines. Publish reproducible results.

## Why InferArena?

Developing new inference algorithms today is slow and difficult.

Researchers often have to:

- fork an inference engine
- modify internal scheduling code
- build custom benchmarks
- collect metrics manually
- reproduce baselines themselves

InferArena separates research from execution.

**Implement your idea once. Evaluate it consistently. Compare it fairly. Share it with others.**

## The old way vs. InferArena

### Today

```
Idea
  ↓
fork vLLM
  ↓
modify scheduler
  ↓
write custom benchmark
  ↓
collect metrics
  ↓
make plots
  ↓
publish
```

### With InferArena

```
Idea
  ↓
class MyScheduler(Scheduler)
  ↓
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,priority,my_scheduler
  ↓
Done.
```

## What you get

```text
                InferArena

          Experiment Definition
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
     Vidur Simulator      Real Engine
                              │
         ┌──────────┬──────────┐
         ▼          ▼          ▼
       vLLM       SGLang    TensorRT
         │
         ▼
      Standard Metrics
         │
         ▼
    Reproducible Report
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run a built-in experiment
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,chunked_prefill,priority,sjf
```

Output:

```text
📈 Throughput +11%
📉 TTFT -8%
💵 Cost/token -6%
📊 GPU utilization +13%

Report saved to: inferarena_outputs/
```

## Add your own strategy

Subclass `Scheduler`, implement `schedule()`, and register it:

```python
from inferarena import Scheduler, Batch


class MyScheduler(Scheduler):
    name = "my_scheduler"

    def schedule(self, state):
        return Batch(requests=state.waiting[:1])
```

Then run it exactly like the built-ins:

```bash
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,my_scheduler
```

See [How to Add a Scheduler](docs/how-to-guides/add-a-scheduler.md) for the full guide.

## Core features

- **Plugin architecture** — implement schedulers, cache policies, and routers as interchangeable components.
- **Fast simulation** — run experiments without GPUs.
- **Real-engine execution** — validate on vLLM, SGLang, and TensorRT-LLM.
- **Standard metrics** — TTFT, latency, throughput, queue time, cache hit rate.
- **Reproducible reports** — JSON summaries, telemetry, plots, and comparison tables.

## Install extras

| Extra | Command | Purpose |
|-------|---------|---------|
| dev | `pip install -e ".[dev]"` | Linting, type checking, tests |
| plot | `pip install -e ".[plot]"` | Telemetry and latency plots |
| dashboard | `pip install -e ".[dashboard]"` | Streamlit web dashboard |
| vllm | `pip install -e ".[vllm]"` | vLLM real-cluster engine |
| sglang | `pip install -e ".[sglang]"` | SGLang real-cluster engine |
| tensorrt | `pip install -e ".[tensorrt]"` | TensorRT-LLM real-cluster engine |
| all | `pip install -e ".[all]"` | All optional backends + plotting + dashboard |

## CLI highlights

```bash
# Compare multiple schedulers on the same workload
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,chunked_prefill,priority,sjf

# Run against a real vLLM deployment
inferarena run --config examples/experiment_vllm.yaml

# Run the built-in benchmark suite
inferarena benchmark

# Launch the web dashboard
inferarena dashboard
```

See [CLI Reference](docs/reference/cli.md) for the complete command list.

## Documentation

- [Getting Started](docs/tutorials/getting-started.md)
- [How to Add a Scheduler](docs/how-to-guides/add-a-scheduler.md)
- [How to Add a Cache Policy](docs/how-to-guides/add-a-cache-policy.md)
- [Architecture](docs/explanation/architecture.md)
- [Deployment](docs/explanation/deployment.md)
- [Web Dashboard](docs/explanation/web-dashboard.md)
- [Design Doc: Why InferArena?](docs/explanation/why-inferarena.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0
