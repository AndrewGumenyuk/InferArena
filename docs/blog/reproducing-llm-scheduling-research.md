# Reproducing LLM scheduling research in under five minutes

Every LLM scheduling paper rebuilds the same infrastructure before it can test a
single idea: fork an inference engine, hack the scheduler, write a benchmark
harness, collect metrics, reimplement the baselines. The idea itself is often
the smallest part of the work.

This post shows the opposite workflow. We'll reproduce the core scheduling
algorithm from **Sarathi-Serve** (OSDI 2024, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310))
as a 74-line plugin, run it against two baselines on a laptop in seconds, and
get a comparison table, plot, and report — no GPU, no engine fork.

## The idea we're reproducing

Naive schedulers run prefill to completion before a request joins the decode
batch. Long prompts block short ones behind them — **head-of-line blocking** —
and decode GPUs sit idle while a big prefill executes.

Sarathi-Serve's insight: **chunk the prefill**. Each step, pack the token budget
with (1) all running decode requests first, then (2) partial prefills of
already-admitted requests, then (3) new requests. Decodes never stall, and a
1024-token prompt no longer monopolizes the engine — it flows through in
budget-sized pieces.

## The entire reproduction

The plugin (`src/inferarena/plugins/schedulers/sarathi_serve.py`) implements
Algorithm 3 from the paper against one interface:

```python
class Scheduler(ABC):
    @abstractmethod
    def schedule(self, state: SystemState) -> Batch:
        """Select requests (and per-request token counts) for the next step."""
```

`SystemState` gives the waiting queue, running requests, and the token budget.
The `Batch` it returns can carry per-request token counts — which is all you
need to express chunked prefill. No engine internals, no CUDA, no vLLM fork.

The experiment is a YAML file:

```yaml
# examples/case_study_variable.yaml
name: variable-prompt-case-study
workload:
  name: variable
  num_requests: 64
  arrival_rate: 4.0
  prompt_tokens_min: 64
  prompt_tokens_max: 1024   # 2x the per-step budget, so chunking matters
  output_tokens_min: 16
  output_tokens_max: 128
  seed: 42
engine:
  max_tokens_per_step: 512
max_steps: 20000
```

And the run:

```bash
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf,sarathi_serve
```

## Results

Same workload, same seed, same 20,000-step budget, three policies:

| Metric | FCFS | SJF | Sarathi-Serve |
|---|---|---|---|
| Completed requests | 2 | 33 | **64** |
| Completion rate | 3.1% | 51.6% | **100%** |
| Throughput (rps) | 0.09 | 0.93 | **1.03** |
| Total steps | 20000 | 20000 | **3023** |
| TTFT p50 (ms) | 23.30 | 38.55 | 22789.03 |

FCFS processes requests in arrival order; a 1024-token prompt eats multiple
prefill steps while short requests pile up behind it. SJF jumps short prompts
ahead — 16x more completions — but starves long ones (31 never finish).
Sarathi-Serve chunks the large prefills, never stalls a decode, and completes
the entire workload in 3,023 steps.

## The honest caveats

A result this dramatic deserves scrutiny, and the framework makes the scrutiny
possible:

- **Survivorship bias.** FCFS's TTFT p50 of 23 ms is computed over the *two*
  requests it finished. Sarathi-Serve's 22.8 s TTFT reflects all 64. Latency
  percentiles across policies with different completion rates are not directly
  comparable — the headline metric here is **completion rate and throughput**,
  not latency.
- **Fairness has a price.** Sarathi-Serve's high TTFT p50 is real: keeping
  every decode alive means new prefills wait. That's the policy's actual
  trade-off, and it shows up in the numbers.
- **Simulation is not silicon.** The timing model is analytical
  (`prefill_time_per_token`, `decode_time_per_token` in the config), not
  measured from a GPU. Treat milliseconds as relative; trust the *logical*
  results (completion counts, queue behavior, scheduling order). The full
  list of what is and isn't modeled is in
  [simulation-assumptions.md](../explanation/simulation-assumptions.md), and
  what we have and haven't validated against real engines is in
  [validation.md](../explanation/validation.md).

## Why this matters

The point isn't that Sarathi-Serve wins a synthetic benchmark — the paper
already showed that on real hardware. The point is that the reproduction costs
**one file and one command**, and every future idea gets the same deal:
implement `schedule()`, run `inferarena compare`, get the same metrics, the
same baselines, the same report format. That's the workflow scheduling research
should have.

Try it:

```bash
git clone https://github.com/AndrewGumenyuk/InferArena.git
cd InferArena && pip install -e ".[dev]"
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf,sarathi_serve
```

Then write your own scheduler — the
[guide](../how-to-guides/add-a-scheduler.md) takes about ten minutes.
