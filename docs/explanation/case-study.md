# Case study: beating head-of-line blocking with shortest-job-first

This case study shows how a one-line scheduling change can dramatically improve throughput when prompt lengths vary and the batch token budget is tight.

## The scenario

Imagine a chat API that accepts requests with very different prompt sizes:

- Some users send short prompts: 64 tokens.
- Other users paste long documents: up to 1024 tokens.
- The serving engine can process at most 512 tokens per step.
- New requests arrive at 4 requests per second.

Under FCFS (first-come-first-served), a long prompt that arrives early monopolizes the batch budget. Shorter prompts that arrive later must wait, even though they could have been served quickly. This is classic *head-of-line blocking*.

## Workload config

The config `examples/case_study_variable.yaml` captures this scenario:

```yaml
name: variable-prompt-case-study
scheduler: fcfs
cache_policy: no_op
workload:
  name: variable
  generator: variable
  num_requests: 64
  arrival_rate: 4.0
  prompt_tokens_min: 64
  prompt_tokens_max: 1024
  output_tokens_min: 16
  output_tokens_max: 128
  seed: 42
engine:
  name: simulation
  max_tokens_per_step: 512
max_steps: 20000
output_dir: ./inferarena_outputs/case_study_variable
```

Key knobs:

- `max_tokens_per_step: 512` creates the tight budget.
- `prompt_tokens_max: 1024` creates requests that do not fit in a single step.
- `generator: variable` produces the prompt-length spread.
- `seed: 42` makes the workload deterministic.

## Running the comparison

Compare FCFS against shortest-job-first (SJF):

```bash
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf
```

## Results

| Scheduler | Completed | Throughput (rps) | TTFT p50 (ms) | TTFT p99 (ms) | Latency p50 (ms) |
|-----------|-----------|------------------|---------------|---------------|------------------|
| fcfs      | 2         | 0.09             | 23.30         | 37.41         | 1457.90          |
| sjf       | 33        | 0.93             | 38.55         | 64.62         | 1595.15          |

SJF completes **16.5x more requests** and achieves **10.3x higher throughput** than FCFS within the same 20,000-step budget.

## Why this happens

FCFS processes requests in arrival order. When a 1024-token prompt arrives first, it consumes multiple prefill steps before any later request can enter the batch. By the time the long request finishes prefill, many short requests are stuck in the queue and the run ends before they are served.

SJF reorders the waiting queue by estimated job size. Short prompts jump ahead, get through prefill quickly, and free the batch budget for the next short prompt. The long prompts still run, but they no longer block the rest of the queue.

## Trade-off

SJF slightly increases per-request TTFT for the long prompts that are deprioritized, and median latency is marginally higher because the scheduler is deliberately favoring throughput over individual latency. The case study is not claiming SJF is universally better; it is showing that the right scheduling policy depends on the workload shape.

## Takeaway

A realistic workload shape plus a constrained batch budget can turn an obvious baseline like FCFS into a poor choice. InferArena makes it trivial to discover this: implement or select a scheduler, run the same workload, and compare the numbers.

Try it yourself:

```bash
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf,chunked_prefill,priority
```
