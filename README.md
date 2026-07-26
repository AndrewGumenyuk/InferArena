# InferArena

> An open-source experimentation platform for LLM inference systems.

InferArena lets researchers and engineers implement schedulers, cache policies, and routers as plugins,
and evaluate them fairly across fast simulation and (eventually) real-cluster execution.

## Why InferArena?

Building a better LLM inference scheduler today usually means forking vLLM and writing custom
scripts. InferArena provides a shared substrate where you write one plugin and get:

- Fast simulation without GPUs.
- Standardized metrics and fair baselines.
- Side-by-side scheduler comparisons.
- Reproducible experiment reports with optional plots.
- A plugin registry to share your work.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install InferArena in editable mode with dev tools
pip install -e ".[dev]"

# 3. Run a built-in experiment
inferarena run --config examples/experiment.yaml
```

The example config runs 32 uniform requests through the FCFS scheduler and writes reports to
`./inferarena_outputs/`.

## Docker

Run InferArena without installing Python locally:

```bash
# Build the image
docker build -t inferarena .

# Run a built-in experiment
docker run --rm -it \
  -v $(pwd)/inferarena_outputs:/app/inferarena_outputs \
  inferarena run --config examples/experiment.yaml

# Or use Docker Compose
docker compose run --rm inferarena run --config examples/experiment.yaml
```

See [Deployment](docs/explanation/deployment.md) for vLLM-backed Compose profiles and Kubernetes notes.

## Install extras

| Extra | Command | Purpose |
|-------|---------|---------|
| dev | `pip install -e ".[dev]"` | Linting, type checking, tests |
| plot | `pip install -e ".[plot]"` | Telemetry and latency plots |
| vllm | `pip install -e ".[vllm]"` | vLLM real-cluster engine |
| sglang | `pip install -e ".[sglang]"` | SGLang real-cluster engine |
| tensorrt | `pip install -e ".[tensorrt]"` | TensorRT-LLM real-cluster engine |
| all | `pip install -e ".[all]"` | All optional backends + plotting |

## CLI

```bash
# List built-in schedulers and cache policies
inferarena list-schedulers
inferarena list-cache-policies

# Run an experiment from a YAML config
inferarena run --config examples/experiment.yaml

# Run a multi-GPU simulation
inferarena run --config examples/experiment_multi_gpu.yaml

# Run against a real vLLM deployment
inferarena run --config examples/experiment_vllm.yaml

# Run against a real SGLang deployment
inferarena run --config examples/experiment_sglang.yaml

# Run against a real TensorRT-LLM deployment
inferarena run --config examples/experiment_tensorrt.yaml

# Compare multiple schedulers on the same workload
inferarena compare --config examples/experiment.yaml --schedulers fcfs,chunked_prefill,priority,sjf

# Run the built-in benchmark suite
inferarena benchmark

# Download a public trace dataset
inferarena download-dataset sharegpt_vicuna --output ./datasets/sharegpt.json

# List known datasets
inferarena list-datasets
```

## Python API

```python
from inferarena import ExperimentRunner, ExperimentSpec, WorkloadSpec, EngineSpec

runner = ExperimentRunner()
spec = ExperimentSpec(
    name="fcfs-baseline",
    scheduler="fcfs",
    workload=WorkloadSpec(
        num_requests=64,
        arrival_rate=2.0,
        prompt_tokens=512,
        output_tokens=128,
        seed=42,
    ),
    engine=EngineSpec(max_tokens_per_step=2048),
)
result = runner.run(spec)
print(result.summary())
```

## Add a scheduler

Subclass `Scheduler`, implement `schedule(state)`, and register it:

```python
from inferarena import Scheduler, Batch


class MyScheduler(Scheduler):
    name = "my_scheduler"

    def schedule(self, state):
        return Batch(requests=state.waiting[:1])
```

See [How to Add a Scheduler](docs/how-to-guides/add-a-scheduler.md) for the full guide.

## Features

- Plugin-based schedulers, cache policies, and routers.
- Built-in schedulers: FCFS, chunked prefill, priority, shortest-job-first, round-robin.
- Built-in cache policies: no-op, exact-prefix cache.
- Built-in routers: round-robin, least-loaded.
- Single-GPU and data-parallel multi-GPU discrete-event simulation.
- Real-cluster vLLM, SGLang, and TensorRT-LLM execution via OpenAI-compatible APIs.
- Synthetic and real-trace workloads (ShareGPT / JSON).
- Dataset downloader for public traces.
- Multi-scheduler comparison reports.
- Benchmark suite with standardized configs.
- Optional telemetry and latency CDF plots.
- Jupyter notebook dashboard for result exploration.
- Docker and Docker Compose support.
- CLI for running, comparing, benchmarking, and downloading datasets.

## Development

```bash
# Run tests
cd /Users/andriihumeniuk/Projects/relay
source .venv/bin/activate
pytest

# Run linters and type checker
ruff check .
ruff format --check .
mypy src/inferarena

# Auto-fix formatting
ruff format .
ruff check . --fix
```

## Documentation

- [Getting Started](docs/tutorials/getting-started.md)
- [How to Add a Scheduler](docs/how-to-guides/add-a-scheduler.md)
- [How to Add a Cache Policy](docs/how-to-guides/add-a-cache-policy.md)
- [Architecture](docs/explanation/architecture.md)
- [vLLM Integration](docs/explanation/vllm-integration.md)
- [SGLang Integration](docs/explanation/sglang-integration.md)
- [TensorRT-LLM Integration](docs/explanation/tensorrt-integration.md)
- [Deployment](docs/explanation/deployment.md)
- [Vidur Integration](docs/explanation/vidur-integration.md)
- [Naming](docs/explanation/naming.md)
- [Dashboard](notebooks/dashboard.ipynb)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0
