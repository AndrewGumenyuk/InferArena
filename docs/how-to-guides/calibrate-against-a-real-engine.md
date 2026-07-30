# How to calibrate the simulator against a real vLLM deployment

This guide runs the experiment that validates InferArena's simulator: measure
real per-token costs on a GPU-backed vLLM server, replay a fixed workload
against it, then check whether the calibrated simulator reproduces the same
trends.

You need a machine with an NVIDIA GPU. You do **not** need a big one — a free
Google Colab T4 works.

## Option A: Google Colab (free, ~30 minutes)

1. Open a new notebook at <https://colab.research.google.com> and select
   **Runtime → Change runtime type → T4 GPU**.

2. Install vLLM and InferArena:

   ```bash
   !pip install -q vllm openai
   !pip install -q "git+https://github.com/AndrewGumenyuk/InferArena.git"
   ```

3. Start vLLM in the background with a small model. Note
   `--max-num-batched-tokens`: the calibration script must use the same value.

   ```bash
   !nohup vllm serve Qwen/Qwen2.5-0.5B \
       --max-num-batched-tokens 2048 \
       --max-num-seqs 64 > vllm.log 2>&1 &
   ```

   Wait until `vllm.log` shows the server is ready (downloading the model the
   first time takes a few minutes):

   ```bash
   !tail -f vllm.log   # stop when you see "Application startup complete"
   ```

4. Run the calibration:

   ```bash
   !python -m scripts.calibrate_against_vllm \
       --base-url http://localhost:8000/v1 \
       --model Qwen/Qwen2.5-0.5B \
       --max-tokens-per-step 2048 \
       --requests 32 --arrival-rate 2.0
   ```

   (If you installed from GitHub rather than a source checkout, download the
   script first:
   `!wget https://raw.githubusercontent.com/AndrewGumenyuk/InferArena/main/scripts/calibrate_against_vllm.py`
   and run `!python calibrate_against_vllm.py ...`.)

5. Read the results:

   ```bash
   !cat inferarena_outputs/calibration/calibration_report.md
   ```

## Option B: rented GPU (RunPod / Lambda / vast.ai, ~$1)

Any instance with an L4/A10/T4 and the `vllm/vllm-openai:latest` Docker image
works. Start the server as above, then run the same command from a machine
that can reach the endpoint (set `--base-url` accordingly).

## What the script does

`scripts/calibrate_against_vllm.py` runs three phases:

1. **Microbenchmark** — fits `prefill_time_per_token` from TTFT vs prompt
   size (exact prompt token counts come from the API's `usage` field) and
   measures `decode_time_per_token` as the median inter-token gap while
   streaming at concurrency 1.
2. **Replay** — runs the same seeded workload (a) against the live server via
   the vLLM adapter and (b) through the simulator configured with the measured
   parameters and the `chunked_prefill` scheduler (the closest InferArena
   policy to vLLM's default).
3. **Compare** — writes `comparison.json` and `calibration_report.md` with
   throughput, TTFT/latency percentiles, and the sim/real ratio for each.

## How to read the results

**Success looks like:** same completion counts, throughput within ~2x, TTFT
and latency distributions with the same shape and ordering. **Failure looks
like:** the simulator ranking policies differently than the real engine, or
errors of 10x+ that can't be explained by the known missing overheads.

Expected, explainable gaps (documented in
[simulation-assumptions.md](../explanation/simulation-assumptions.md)):

- Real TTFT includes network, tokenizer, and scheduling overhead — visible as
  the **prefill intercept** in the report. The simulator has no fixed
  overhead term.
- Real decode steps slow down as the batch grows; the simulator's
  `decode_time_per_token` is flat (measured at concurrency 1). Expect the
  simulator to be optimistic at high load.

## Share your results

Whatever you find — match or divergence — open an issue with your
`comparison.json`, the GPU/model you used, and the vLLM version. Divergence is
as valuable as agreement: it tells us where the timing model needs work.
