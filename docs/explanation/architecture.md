# Architecture

InferArena is built around a small harness pattern:

```text
                InferArena

          Experiment Definition
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
     Vidur Simulator      Real Engine
                              │
         ┌──────────┬──────────┐
         ▼          ▼          ▼
       vLLM       SGLang    TensorRT
         │
         ▼
      Standard Metrics
         │
         ▼
    Reproducible Report
```

- **Inference components** implement a decision strategy: scheduler, cache policy, router, and future component types.
- **ExperimentSpec** declares what to run.
- **ExecutionEngine** abstracts simulation, emulation, and real-cluster runs.
- **ExperimentRunner** orchestrates the full pipeline.
- **MetricsCollector** and **ReportGenerator** capture and emit results.

This separation keeps components thin and the harness reusable.

## Inference components

An inference component is any pluggable decision in the inference pipeline.

```text
InferenceComponent
       │
       ├── Scheduler
       ├── CachePolicy
       ├── Router
       └── (future: BatchPolicy, KVPolicy, MoEPolicy, ...)
```

Each component inherits from a small base class and implements a single method. For example, a scheduler implements `schedule(state) -> Batch`. This lets researchers focus on the idea rather than the surrounding engine code.

## Engines

Engines execute the workload. The same component can be evaluated across multiple engines.

- `SimulationEngine`: fast single-GPU discrete-event simulation.
- `MultiGPUSimulationEngine`: data-parallel simulation across multiple workers.
- `VLLMEngine`: real-cluster execution via vLLM's OpenAI-compatible API.
- `SGLangEngine`: real-cluster execution via SGLang's OpenAI-compatible API.
- `TensorRTEngine`: real-cluster execution via TensorRT-LLM's OpenAI-compatible API.
- `VidurEngine`: planned adapter for high-fidelity Vidur simulation.

## Execution flow

1. The runner loads the `ExperimentSpec` from YAML.
2. It instantiates the selected inference component plugins.
3. It creates the execution engine declared in the spec.
4. The engine runs the workload and emits per-request results.
5. The metrics collector aggregates results.
6. The report generator writes `summary.json`, `telemetry.jsonl`, `requests.json`, optional plots, and markdown reports.
