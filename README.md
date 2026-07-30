# InferArena

[![CI](https://github.com/AndrewGumenyuk/InferArena/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewGumenyuk/InferArena/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> The open-source experimentation platform for LLM inference systems.
>
> Implement an inference strategy once. Evaluate it consistently across simulation and production inference engines. Compare against standard baselines. Publish reproducible results.

![InferArena demo: comparing FCFS, SJF, and Sarathi-Serve](docs/assets/inferarena-demo.gif)

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

## Who is this for?

- **Inference researchers** who want to test a new scheduling, caching, or routing idea without forking vLLM.
- **Systems engineers** evaluating whether a policy change is worth porting to production.
- **Students and teams** learning how LLM serving works through fast, reproducible experiments.

If you are debugging a live production outage, use your production engine's native tooling instead.

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
              Experiment
                  │
                  ▼
        Inference Components
  (Scheduler / Cache / Router)
                  │
                  ▼
         Execution Engine
     ┌────────────┴────────────┐
     ▼                         ▼
 Simulation                Real Engine
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                  vLLM      SGLang    TensorRT
                  │
                  ▼
          Metrics & Reports
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run a built-in experiment
inferarena run --config examples/experiment.yaml
```

Output:

```text
Scheduler: fcfs
Completed: 32
             Metrics
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric             ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ scheduler          │ fcfs     │
│ completed_requests │ 32       │
│ total_steps        │ 701      │
│ total_time_ms      │ 15497.23 │
│ throughput_rps     │ 2.06     │
│ ttft_p50_ms        │ 60.75    │
│ ttft_p99_ms        │ 95.92    │
│ latency_p50_ms     │ 2782.74  │
│ latency_p99_ms     │ 2948.91  │
│ queue_time_p50_ms  │ 9.55     │
│ tbt_p50_ms         │ 21.22    │
│ prefill_p50_ms     │ 51.2     │
│ cache_hits         │ 0        │
│ cache_lookups      │ 16384    │
│ cache_hit_rate     │ 0.0      │
└────────────────────┴──────────┘

Report saved to: inferarena_outputs
```

Compare schedulers on the same workload:

```bash
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,chunked_prefill,priority,sjf
```

## See it optimize something real

A variable-prompt workload with a tight token budget creates head-of-line blocking. FCFS completes 2 requests; shortest-job-first completes 33; Sarathi-Serve completes all 64.

```bash
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf,sarathi_serve
```

![FCFS vs SJF vs Sarathi-Serve on variable prompt lengths](docs/assets/case-study-comparison.png)

| Scheduler | Completed | Throughput (rps) |
|-----------|-----------|------------------|
| FCFS      | 2         | 0.09             |
| SJF       | 33        | 0.93             |
| Sarathi-Serve | 64    | 1.03             |

[Read the case study →](docs/explanation/case-study.md)

## Try it with a real-engine demo

InferArena can evaluate the same strategy against real inference engines through their OpenAI-compatible APIs. To see this without a GPU, run the included mock server demo:

```bash
# Terminal 1: start the mock engine
python scripts/mock_openai_server.py --port 8000

# Terminal 2: run InferArena against it
inferarena compare --config examples/experiment_demo.yaml \
  --schedulers fcfs,sjf
```

Or use Docker Compose:

```bash
docker compose -f docker-compose.demo.yml up --build
```

See [From simulation to a real engine](docs/tutorials/simulation-to-real-engine.md) for the full tutorial.

## What you can do today

- **Run scheduler experiments without a GPU.** The simulation engine runs deterministic workloads in seconds on a laptop.
- **Compare baselines out of the box.** FCFS, chunked prefill, priority, shortest-job-first, and round-robin are built in.
- **Test a custom scheduler in under 50 lines of Python.** Implement one method, register it, and run it against the built-ins.
- **Evaluate cache policies.** Compare no-op and exact-prefix caching on synthetic or trace workloads.
- **Validate on real engines.** Switch the engine to vLLM, SGLang, or TensorRT-LLM and run the same strategy through OpenAI-compatible APIs.
- **Generate reports and plots.** Every run writes JSON summaries, per-request traces, telemetry, comparison tables, and optional latency CDFs.
- **Explore results in a dashboard.** Launch `inferarena dashboard` to browse experiments and compare latency distributions.

InferArena is a research accelerator, not a production serving system. Use it to decide whether an idea is worth porting into vLLM or another engine.

## Add your own strategy

A scheduler is a small class with one method. Here is a greedy shortest-job-first scheduler that packs the batch by estimated prefill cost:

```python
from inferarena.core.batch import Batch
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class GreedyScheduler(Scheduler):
    name = "greedy"

    def schedule(self, state: SystemState) -> Batch:
        batch = Batch()
        used = 0
        for request in state.running + state.waiting:
            cost = 1 if request.is_prefill_complete else request.prompt_tokens
            if used + cost <= state.budget.max_tokens:
                batch.requests.append(request)
                used += cost
        return batch
```

Register it, then run it against built-in baselines:

```bash
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,greedy
```

See [How to Add a Scheduler](docs/how-to-guides/add-a-scheduler.md) for the full guide.

## Core features

- **Plugin architecture** — implement schedulers, cache policies, and routers as interchangeable components.
- **Fast simulation** — run experiments without GPUs.
- **Real-engine execution** — validate on vLLM, SGLang, and TensorRT-LLM.
- **Standard metrics** — TTFT, latency, throughput, queue time, cache hit rate.
- **Reproducible reports** — JSON summaries, telemetry, plots, and comparison tables.

## Project status

InferArena is early but functional. The simulation engine, plugin registry, CLI, reporting pipeline, and built-in schedulers/cache policies are implemented and tested. Real-cluster adapters for vLLM, SGLang, and TensorRT-LLM are available via optional extras. A Vidur-based high-fidelity simulator is on the roadmap.

## How this fits in

| Tool | What it does | InferArena's role |
|------|--------------|-------------------|
| vLLM / SGLang / TensorRT-LLM | Production inference engines. | InferArena plugs into them, but you do not modify their internals to test an idea. |
| vLLM benchmark | Measures a single engine on a fixed workload. | InferArena lets you swap strategies and compare them on the same workload, then move the best one to the engine. |
| Vidur | High-fidelity simulator for LLM clusters. | InferArena is a lighter research harness today; a Vidur adapter is planned for higher-fidelity simulation. |
| MLPerf | Standardized inference benchmark suite. | InferArena is for experimenting with new strategies, not for publishing official benchmark scores. |

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
- [Core Concepts](docs/explanation/concepts.md)
- [Example Configs](examples/README.md)
- [Case Study: beating head-of-line blocking](docs/explanation/case-study.md)
- [How to Add a Scheduler](docs/how-to-guides/add-a-scheduler.md)
- [How to Add a Cache Policy](docs/how-to-guides/add-a-cache-policy.md)
- [How to Add a Router](docs/how-to-guides/add-a-router.md)
- [How to Calibrate Against a Real Engine](docs/how-to-guides/calibrate-against-a-real-engine.md)
- [Architecture](docs/explanation/architecture.md)
- [Simulation Assumptions and Limitations](docs/explanation/simulation-assumptions.md)
- [Validation and Fidelity](docs/explanation/validation.md)
- [Deployment](docs/explanation/deployment.md)
- [Web Dashboard](docs/explanation/web-dashboard.md)
- [Jupyter Dashboard](notebooks/dashboard.ipynb)
- [Design Doc: Why InferArena?](docs/explanation/why-inferarena.md)
- [Blog: Reproducing LLM scheduling research in under five minutes](docs/blog/reproducing-llm-scheduling-research.md)

## Troubleshooting

**No module named 'streamlit'**

Install the dashboard extra:

```bash
pip install -e ".[dashboard]"
```

**No module named 'openai'**

Real-cluster engines require the OpenAI client. Install the relevant extra:

```bash
pip install -e ".[vllm]"   # or [sglang], [tensorrt], [all]
```

**Docker daemon not running**

On macOS you can use [Colima](https://github.com/abiosoft/colima):

```bash
brew install colima docker-buildx docker-compose
colima start
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0
