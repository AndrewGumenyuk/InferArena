# Simulation assumptions and limitations

InferArena's `SimulationEngine` is a lightweight **discrete-event simulator** for
comparing *scheduling decisions*, not a cycle-accurate GPU model. This document
states exactly what it models, what it approximates, and what it does not model —
so you know which numbers to trust.

## What one "step" is

One step = **one scheduler iteration = one batched forward pass**.

At each step the engine:

1. Admits requests whose `arrival_time <= current_time` into the waiting queue.
2. Calls `scheduler.schedule(state)` with a `SystemState` snapshot
   (waiting, running, completed, `TokenBudget`).
3. Executes the returned `Batch`: each request advances by its per-request token
   count (`Batch.token_counts`, defaulting to the full remaining prefill or
   1 decode token).
4. Advances simulated time by the step duration.

`max_tokens_per_step` (default 2048) is the **token budget per step** — the same
abstraction Sarathi-Serve and Orca use. Prompts larger than the budget are
processed incrementally across multiple steps (**chunked prefill**).

## Request lifecycle

```
WAITING → RUNNING (first time scheduled)
        → prefill (one or more chunks, until prefilled_tokens >= prompt_tokens)
        → decode (1 token per step per request, unless the scheduler says otherwise)
        → COMPLETED (generated_tokens >= max_output_tokens)
```

A running request that is *not* included in a batch is simply paused for that
step; it keeps its progress and stays in the running set.

## Where the timing numbers come from

Timing is **analytical and fully configurable** — it is *not* measured from real
hardware. The duration of a step is the maximum over the requests in the batch:

| Request in batch | Contribution to step time |
|---|---|
| Prefill (chunk of *n* tokens) | `prefill_time_per_token × (n − cached_tokens)` |
| Decode | `decode_time_per_token` (flat, per step) |

Defaults in `EngineSpec`: `prefill_time_per_token = 0.1 ms`,
`decode_time_per_token = 20 ms`. **These are synthetic placeholders.** An empty
step takes a 1 ms floor. All times are in milliseconds.

Modeling a batch as "the slowest member sets the step time" approximates one
forward pass reasonably well, and it correctly captures the key chunked-prefill
effect: a decode request sharing a step with a large prefill chunk is delayed by
that chunk.

## What TTFT includes in simulation

```
TTFT = queue time (arrival → first scheduled, including head-of-line blocking)
     + prefill execution (across all chunks)
```

Not included: network, tokenization/detokenization, sampling, API/server
overhead, KV-cache transfer.

## What is NOT modeled

Be aware of these before drawing conclusions from absolute numbers:

- **No memory model.** There is no KV-cache capacity limit, so there is no
  preemption, eviction, or OOM-driven recomputation. (`RequestStatus.PREEMPTED`
  exists but no engine path uses it yet.)
- **No max batch size.** Only the token budget constrains a batch.
- **Batch-size-independent decode time.** Real decode steps get slower as the
  batch grows (until compute-bound); the simulator charges a flat
  `decode_time_per_token`.
- **Uniform per-token costs.** No long-context attention slowdown, no MoE
  routing cost, no speculative decoding.
- **No preemption cost.** A scheduler can "pause" a running request by not
  scheduling it, but pausing is free — real preemption costs recompute or
  swap-out.
- **Cache metric quirk.** `cache_lookups` is incremented by `prompt_tokens` on
  *every* prefill step, so chunked prefills count lookups multiple times.
- **Censored requests.** Requests that don't finish within `max_steps` have no
  completion metrics; all latency/TTFT percentiles are computed **over completed
  requests only** (survivorship bias — always compare `completed_requests`
  alongside latency numbers).

## Which metrics to trust

**Logical metrics (trust these):**

- completed / unfinished request counts, completion rate
- scheduling order and queue depth over time
- steps taken, token budget utilization
- cache hit rate, fairness and starvation behavior

**Performance metrics (treat as *relative*, not absolute):**

- TTFT / TBT / end-to-end latency in milliseconds
- throughput in requests/second
- any cost-per-token derived from them

Millisecond numbers are only as meaningful as the `prefill_time_per_token` /
`decode_time_per_token` you supply. With calibrated values, simulated *trends*
(policy rankings, distribution shapes) should track reality; absolute values may
still diverge.

## How to calibrate

1. Measure real per-token costs for your model and GPU (e.g. run
   `vllm bench` or time a small serve):
   - prefill ms/token at your typical prompt sizes,
   - decode ms/token at your typical batch sizes.
2. Set them in your experiment YAML:

   ```yaml
   engine:
     max_tokens_per_step: 2048
     prefill_time_per_token: 0.05   # measured
     decode_time_per_token: 12.0    # measured
   ```

3. Compare **trends** (does policy A beat policy B? by roughly how much?)
   against a real deployment — see
   [Validation and fidelity](validation.md) for what we've checked so far.
