# CLI Reference

## `inferarena run`

Run an experiment from a YAML config.

```bash
inferarena run --config examples/experiment.yaml
inferarena run --config examples/experiment_multi_gpu.yaml
inferarena run --config examples/experiment_vllm.yaml
```

## `inferarena compare`

Run the same workload with multiple schedulers and print a comparison table.

```bash
inferarena compare --config examples/experiment.yaml --schedulers fcfs,chunked_prefill,priority,sjf
```

If `--schedulers` is omitted, all registered schedulers are compared.

## `inferarena benchmark`

Run all benchmark configs in a directory and aggregate results.

```bash
inferarena benchmark
inferarena benchmark --directory ./my_benchmarks
```

## `inferarena download-dataset`

Download a public trace dataset by alias or URL.

```bash
inferarena download-dataset sharegpt_vicuna --output ./datasets/sharegpt.json
```

## `inferarena list-datasets`

List known downloadable dataset aliases.

```bash
inferarena list-datasets
```

## `inferarena list-schedulers`

List registered scheduler plugins.

```bash
inferarena list-schedulers
```

## `inferarena list-cache-policies`

List registered cache policy plugins.

```bash
inferarena list-cache-policies
```
