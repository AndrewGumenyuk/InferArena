# From simulation to a real engine

This tutorial shows the full InferArena workflow: design a strategy in fast simulation, then validate it against a real engine's API surface.

You do not need a GPU for this tutorial. We use a mock OpenAI-compatible server that behaves like vLLM's API without loading a model.

## What you will learn

1. Run a scheduler comparison in simulation.
2. Switch the engine from simulation to a real-engine adapter.
3. Run the same strategy against the mock server.
4. Compare simulation and real-engine outputs.

## Step 1: run in simulation

Start with the built-in variable-prompt case study:

```bash
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf
```

This takes seconds on a laptop and produces a comparison in `inferarena_outputs/case_study_variable/comparison/`.

## Step 2: inspect the simulation config

Open `examples/case_study_variable.yaml`. The engine block is:

```yaml
engine:
  name: simulation
  max_tokens_per_step: 512
```

The simulator is deterministic and fast, but it is still a model. The next step is to see whether the same scheduler behaves sensibly when it talks to an external engine.

## Step 3: switch to a real-engine adapter

Create a new config that points to the mock server:

```yaml
name: "demo-real-engine"
scheduler: "fcfs"
cache_policy: "no_op"
workload:
  name: "uniform"
  num_requests: 8
  arrival_rate: 1.0
  prompt_tokens: 128
  output_tokens: 16
  seed: 42
engine:
  name: "vllm"
  model: "mock-model"
  base_url: "http://localhost:8000/v1"
  api_key: "dummy"
max_steps: 10000
output_dir: "./inferarena_outputs/demo_real_engine"
```

This config is saved as `examples/experiment_demo.yaml`.

## Step 4: start the mock server

In one terminal:

```bash
python scripts/mock_openai_server.py --port 8000
```

In another terminal, run InferArena:

```bash
inferarena run --config examples/experiment_demo.yaml
```

The engine sends OpenAI-compatible chat completion requests to the mock server, streams the response, and records TTFT and latency.

## Step 5: run the same comparison against the mock server

```bash
inferarena compare --config examples/experiment_demo.yaml \
  --schedulers fcfs,sjf
```

You now have the same scheduler evaluation pattern as in simulation, but the requests travel over HTTP to an external service.

## Step 6: compare with Docker Compose

If you prefer a one-command demo, use Docker Compose:

```bash
docker compose -f docker-compose.demo.yml up --build
```

This starts the mock server and runs the demo experiment in linked containers.

## What to expect

- Simulation gives you fast, reproducible numbers for algorithm selection.
- The real-engine adapter proves the strategy can be plugged into a production engine with the same interface.
- The mock server demonstrates the API integration. On a Linux machine with a GPU, replace the mock server with `vllm serve` and use `examples/experiment_vllm.yaml`.

## Next steps

- Read the [vLLM integration guide](../explanation/vllm-integration.md) to run against a real vLLM deployment.
- Read [How to Add a Scheduler](../how-to-guides/add-a-scheduler.md) to implement your own strategy.
- Run the [variable-prompt case study](../explanation/case-study.md) to see a realistic optimization example.
