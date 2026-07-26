# SGLang Integration

InferArena can execute experiments against a live [SGLang](https://github.com/sgl-project/sglang) deployment through its OpenAI-compatible API.

## Install

```bash
pip install -e ".[sglang]"
```

The SGLang extra pulls in the SGLang runtime. The OpenAI client is also required and is included transitively.

## Start SGLang

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --port 30000
```

## Run an experiment

```bash
inferarena run --config examples/experiment_sglang.yaml
```

The example config points to `http://localhost:30000/v1` and uses the default model name.

## Engine spec fields

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `default` | Model name exposed by SGLang. |
| `base_url` | `http://localhost:30000/v1` | SGLang OpenAI-compatible endpoint. |
| `api_key` | `dummy` | Optional API key. |

## Limitations

- The scheduler plugin is used for reporting only; SGLang controls its own scheduling at runtime.
- Cache policies are not enforced against the live server.
- Network errors are logged and the partial result is retained.
