# Architecture

InferArena separates inference decisions from execution. Researchers implement small, pluggable **inference components** while the framework handles workloads, execution, metrics, and reporting across simulation and production engines.

```text
              Experiment
                  │
                  ▼
        Inference Components
  (Scheduler / Cache / Router)
                  │
                  ▼
         Execution Engine
     ┌────────────┴────────────┐
     ▼                         ▼
 Simulation                Real Engine
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                  vLLM      SGLang    TensorRT
                  │
                  ▼
          Metrics & Reports
```

## Inference components

An inference component is a small, pluggable decision module that modifies one aspect of the inference pipeline while everything else remains fixed.

```text
InferenceComponent
       │
       ├── Scheduler
       ├── CachePolicy
       ├── Router
       └── (future: BatchPolicy, KVPolicy, MoEPolicy, ...)
```

Each component inherits from a small base class and implements a single, component-specific method. A scheduler implements `schedule(state) -> Batch`. A cache policy implements `lookup(request)` and `store(request)`. A router picks a replica for an incoming request.

This lets researchers focus on the idea rather than the surrounding engine code.

## Engines

Engines execute the workload. The same component can be evaluated across multiple engines.

| Engine | Purpose |
|---|---|
| `SimulationEngine` | Fast deterministic experiments on a laptop. |
| `MultiGPUSimulationEngine` | Simulate multiple GPUs and routing between them. |
| `VLLMEngine` | Execute against a real vLLM deployment. |
| `SGLangEngine` | Execute against a real SGLang deployment. |
| `TensorRTEngine` | Execute against a real TensorRT-LLM deployment. |
| `VidurEngine` | Planned adapter for high-fidelity Vidur simulation. |

## Execution flow

1. **Define an experiment** in a YAML config.
2. **Select inference components** to evaluate.
3. **Choose an execution engine**: simulation or real cluster.
4. **Run the workload**.
5. **Collect standardized metrics**.
6. **Generate reproducible reports**.

```text
experiment.yaml
        │
        ▼
ExperimentRunner
        │
        ▼
Inference Components
        │
        ▼
Execution Engine
        │
        ▼
Metrics
        │
        ▼
Report
```

## Design philosophy

InferArena separates decision logic (what policy to use) from execution (where it runs). Researchers implement small inference components while the framework handles workloads, execution, metrics, and reporting. This makes experiments easier to reproduce and compare across engines.

The scheduler plugin is the star. The engine is interchangeable.
