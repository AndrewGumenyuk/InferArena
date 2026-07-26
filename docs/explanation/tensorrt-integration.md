# TensorRT-LLM Integration

InferArena can execute experiments against a live [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) deployment through an OpenAI-compatible front-end, typically served via Triton Inference Server.

## Install

```bash
pip install -e ".[tensorrt]"
```

The TensorRT-LLM extra pulls in the Python runtime. The OpenAI client is required and included transitively.

## Start TensorRT-LLM

Refer to the TensorRT-LLM documentation for building and launching an engine for your target model. A common pattern uses Triton with the TensorRT-LLM backend and an OpenAI-compatible HTTP front-end on port `8000`.

## Run an experiment

```bash
inferarena run --config examples/experiment_tensorrt.yaml
```

The example config points to `http://localhost:8000/v2` and uses the default model name `ensemble`.

## Engine spec fields

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `ensemble` | Model name exposed by the server. |
| `base_url` | `http://localhost:8000/v2` | TensorRT-LLM OpenAI-compatible endpoint. |
| `api_key` | `dummy` | Optional API key. |

## Limitations

- The scheduler plugin is used for reporting only; TensorRT-LLM controls its own scheduling at runtime.
- Cache policies are not enforced against the live server.
- Network errors are logged and the partial result is retained.
