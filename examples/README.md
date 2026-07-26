# InferArena Examples

This directory contains example experiment configs. Each config is a YAML file that declares a workload, scheduler, cache policy, engine, and output directory.

## Simulation examples

| File | What it shows |
|---|---|
| `experiment.yaml` | Basic FCFS simulation with a uniform workload. Start here. |
| `experiment_multi_gpu.yaml` | Data-parallel multi-GPU simulation with a router. |
| `experiment_prefix_cache.yaml` | Exact-prefix cache policy simulation. |
| `experiment_trace.yaml` | Load requests from a ShareGPT-style trace file. |

## Real-cluster examples

| File | What it shows |
|---|---|
| `experiment_vllm.yaml` | Run against a live vLLM deployment. |
| `experiment_sglang.yaml` | Run against a live SGLang deployment. |
| `experiment_tensorrt.yaml` | Run against a live TensorRT-LLM deployment. |

## Running an example

```bash
inferarena run --config examples/experiment.yaml
```

## Comparing schedulers

```bash
inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,chunked_prefill,priority,sjf
```

## Trace data

`sharegpt_sample.jsonl` is a small sample trace file used by `experiment_trace.yaml`. You can download the full ShareGPT dataset with:

```bash
inferarena download-dataset sharegpt_vicuna --output ./datasets/sharegpt.json
```
