# Validation and fidelity

This page answers the hardest question about InferArena honestly: **how much
control do you actually have in each execution mode, and has the simulator been
validated against a real engine?**

## The three execution modes

InferArena has (or plans) three distinct levels of execution. They are **not**
interchangeable, and the difference matters:

| Mode | What InferArena controls | Status |
|---|---|---|
| **Simulation** | The full scheduling decision loop: batch composition, per-request token counts, chunked prefill, admission order | Available |
| **API replay** | The *workload*: request timing, prompt sizes, output lengths. Latency is measured from the outside | Available (vLLM, SGLang, TensorRT-LLM adapters) |
| **Native engine integration** | The engine's *internal* scheduler, via engine-side hooks or patches | Planned |

The critical clarification: **an InferArena scheduler plugin does not run inside
vLLM today.** The OpenAI-compatible API exposes requests and responses, not the
scheduling loop. When you run `engine: vllm`, the adapter replays the workload
against the deployment and measures TTFT/latency from the client side — the
plugin's name is used for labeling and reporting, not runtime control (the
adapter docstrings say this explicitly).

So:

- **Simulation** answers: *"What does this scheduling policy do?"*
- **API replay** answers: *"What latency does this deployment deliver under this
  workload?"*
- **Native integration** will answer: *"Does this policy hold up inside a real
  engine?"* — that requires a deeper engine extension and is on the roadmap.

## What we validated so far

`scripts/validate_simulation.py` runs the **same workload** through both
available modes (8 requests, variable prompt lengths, 0.5 req/s):

| | Simulation (FCFS) | Mock OpenAI server |
|---|---|---|
| Completed requests | **2 / 8** | **8 / 8** |
| Throughput | 0.08 rps | 0.35 rps |
| TTFT p50 | 21.9 ms | 2318 ms |
| What it measures | Scheduling decisions under a token budget | End-to-end API latency, no scheduling model |

The numbers disagree because the two modes **measure different things**. The
simulator's FCFS policy suffers head-of-line blocking under a 512-token step
budget and finishes only 2 requests in 20,000 steps; the mock server has no
scheduler at all and simply answers every request. This comparison demonstrates
the *semantic gap* between the modes — it is **not** a calibration of the
simulator.

## What we have NOT done yet

- **No calibration against real vLLM.** We have not yet run the same workload
  through the simulator and a real GPU-backed vLLM deployment and compared
  policy rankings or latency distributions. Until that exists, treat simulated
  millisecond numbers as relative, not absolute — see
  [Simulation assumptions and limitations](simulation-assumptions.md).
- **No native scheduler execution.** Implementing an InferArena policy inside
  vLLM/SGLang requires engine-side hooks; the current adapters are replay-only.

## What real validation will look like

The bar we intend to hit before calling the simulator "validated":

1. Serve a small model with vLLM (its default scheduler is FCFS-like with
   chunked prefill — a natural baseline).
2. Replay a fixed workload via the vLLM adapter; record TTFT/throughput.
3. Calibrate `prefill_time_per_token` / `decode_time_per_token` from the same
   deployment; run the same workload in simulation.
4. Compare **trends**: do policies rank the same way? Are distribution shapes
   similar? Absolute millisecond parity is not the goal — decision fidelity is.
5. Publish the results, including where the simulator diverges.

`scripts/calibrate_against_vllm.py` automates exactly this — see
[How to calibrate against a real engine](../how-to-guides/calibrate-against-a-real-engine.md).
The script is ready; it has not yet been run against GPU hardware (contributions
of results are welcome).

If you run this comparison yourself, we'd genuinely like the data — open an
issue with your config and measurements.
