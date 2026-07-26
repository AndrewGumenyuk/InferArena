# Architecture

InferArena is built around a harness pattern:

- **Plugins** implement schedulers, cache policies, and routers.
- **ExperimentSpec** declares what to run.
- **ExecutionEngine** abstracts simulation, emulation, and real-cluster runs.
- **ExperimentRunner** orchestrates the full pipeline.
- **MetricsCollector** and **ReportGenerator** capture and emit results.

This separation keeps plugins thin and the harness reusable.

## Plugin types

- **Schedulers** decide which waiting and running requests execute in the next step.
- **Cache policies** decide how many prompt tokens are already cached, reducing prefill work.
- **Routers** decide which worker handles an incoming request in multi-GPU setups.
- **Engines** execute the workload (simulation, multi-GPU simulation, vLLM, Vidur, etc.).

## Engines

- `SimulationEngine`: fast single-GPU discrete-event simulation.
- `MultiGPUSimulationEngine`: data-parallel simulation across multiple workers.
- `VLLMEngine`: planned adapter for real vLLM deployments.
- `VidurEngine`: planned adapter for high-fidelity Vidur simulation.
