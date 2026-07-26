# Getting Started with InferArena

This guide walks you through your first InferArena experiment in about five minutes.

## What you will do

1. Install InferArena in a virtual environment.
2. Run a built-in experiment.
3. Compare two schedulers on the same workload.
4. Inspect the generated report.
5. Launch the web dashboard to explore results.

## Install

```bash
git clone https://github.com/AndrewGumenyuk/InferArena.git
cd InferArena
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you also want plots and the web dashboard, install:

```bash
pip install -e ".[dev,dashboard]"
```

## Run your first experiment

InferArena includes an example config that runs 32 uniform requests through the FCFS scheduler:

```bash
inferarena run --config examples/experiment.yaml
```

You should see output similar to:

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

The report directory contains:

- `summary.json` — key metrics in JSON.
- `requests.json` — per-request results.
- `telemetry.jsonl` — time-series data.
- `report.md` — a human-readable markdown summary.

## Compare schedulers

The main value of InferArena is comparing strategies on the same workload. Run:

```bash
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,chunked_prefill
```

Output:

```text
                              Scheduler Comparison
┏━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━━┳━━━━┳━━━━━┳━━━━┳━━━━━┳━━━━┳━━━━━┓
┃ s… ┃ c… ┃ t… ┃ t… ┃ t… ┃ t… ┃ t… ┃ l… ┃ la… ┃ q… ┃ tb… ┃ p… ┃ ca… ┃ c… ┃ ca… ┃
┡━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━━╇━━━━╇━━━━━╇━━━━╇━━━━━╇━━━━╇━━━━━┩
│ f… │ 32 │ 7… │ 1… │ 2… │ 6… │ 9… │ 2… │ 29… │ 9… │ 21… │ 5… │ 0   │ 1… │ 0.0 │
│ c… │ 32 │ 7… │ 1… │ 2… │ 6… │ 9… │ 2… │ 29… │ 9… │ 21… │ 5… │ 0   │ 1… │ 0.0 │
└────┴────┴────┴────┴────┴────┴────┴────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Comparison report saved to: inferarena_outputs/comparison
```

You can list all built-in schedulers:

```bash
inferarena list-schedulers
```

Output:

```text
Available schedulers:
  - chunked_prefill
  - fcfs
  - priority
  - round_robin
  - sjf
```

## Explore results in the dashboard

If you installed the dashboard extra, launch Streamlit:

```bash
inferarena dashboard
```

Open the URL printed in the terminal. The dashboard shows a summary table, latency CDF comparison, and raw JSON summaries for every experiment in `inferarena_outputs/`.

## Next steps

- [Add a custom scheduler](../how-to-guides/add-a-scheduler.md)
- [Add a custom cache policy](../how-to-guides/add-a-cache-policy.md)
- [Read the architecture overview](../explanation/architecture.md)
- [Run a real-cluster experiment with vLLM](../explanation/vllm-integration.md)
