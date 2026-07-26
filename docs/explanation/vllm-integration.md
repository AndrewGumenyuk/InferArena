# vLLM Integration

InferArena can run experiments against a real [vLLM](https://github.com/vllm-project/vllm) deployment through its OpenAI-compatible API.

## Install

```bash
pip install -e ".[vllm]"
```

## Start vLLM

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --max-num-seqs 256
```

## Run an experiment

```bash
inferarena run --config examples/experiment_vllm.yaml
```

## How it works

- The engine loads the workload and spawns API calls according to each request's arrival time.
- It streams the response and records:
  - `first_token_time` (TTFT)
  - `completion_time` (end-to-end latency)
- Requests run concurrently via `asyncio`, matching how a real client would submit traffic.

## Configuration

Engine spec fields:

| Field | Default | Description |
|---|---|---|
| `model` | `vllm-model` | Model name served by vLLM |
| `base_url` | `http://localhost:8000/v1` | vLLM API endpoint |
| `api_key` | `dummy` | API key if authentication is enabled |

## Caveats

- The engine maps `prompt_tokens` to a dummy prompt of roughly that length. It does not use real tokenizers.
- Failures are logged but do not stop the experiment; partial results are kept.
