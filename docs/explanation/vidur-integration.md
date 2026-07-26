# Vidur Integration

[Vidur](https://arxiv.org/pdf/2405.05465) is a high-fidelity, large-scale simulator for LLM inference developed by Georgia Tech and Microsoft Research. It profiles operators, builds runtime estimators, and runs an event-driven simulation that can predict latency within ~9% of real deployments.

InferArena and Vidur solve different but complementary problems:

| Concern | InferArena | Vidur |
|---|---|---|
| Primary goal | Fair scheduler/plugin evaluation | High-fidelity deployment simulation |
| Speed | Very fast, GPU-free | Slower, model- and hardware-aware |
| Plugin model | Thin `Scheduler.schedule(state)` | Pluggable hierarchical scheduler |
| Workloads | Synthetic + trace loaders | Vidur-Bench curated traces |
| Output | Comparative metrics + reports | Capacity planning + config optimizer |

Because of this split, InferArena can use Vidur as a downstream **high-fidelity validation engine** while keeping its own fast simulation for rapid scheduler iteration.

## Proposed integration architecture

```
                InferArena experiment spec
                        │
                        ▼
            ┌───────────────────────┐
            │    ExperimentRunner   │
            └───────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   FastSimulation   VidurEngine    RealCluster
   (iteration)      (validation)   (production)
```

1. **Fast iteration** in InferArena's own simulator.
2. **Promising plugins** are re-run through a `VidurEngine` adapter for hardware-accurate numbers.
3. **Reports** from both engines are stored together for comparison.

## Adapter sketch

A future `VidurEngine` would:

- Translate InferArena's `WorkloadSpec` into a Vidur request trace.
- Map InferArena's `Scheduler` plugin onto Vidur's scheduler interface.
- Run Vidur's simulator with a chosen model/GPU configuration.
- Convert Vidur's per-request metrics into InferArena's `ExperimentResult`.

```python
class VidurEngine(ExecutionEngine):
    name = "vidur"

    def __init__(self, scheduler: Scheduler, engine_spec: EngineSpec) -> None:
        self.scheduler = scheduler
        self.engine_spec = engine_spec

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        vidur_trace = to_vidur_trace(spec.workload)
        vidur_scheduler = VidurSchedulerAdapter(self.scheduler)
        vidur_result = vidur.simulate(
            trace=vidur_trace,
            scheduler=vidur_scheduler,
            model=spec.engine.model_name,
            gpu=spec.engine.gpu_name,
        )
        return from_vidur_result(vidur_result, self.scheduler.name)
```

## Open questions

- Vidur is not currently published as a stable pip-installable package. The adapter would need to depend on a pinned Git revision or a Vidur-provided API.
- Scheduler semantics differ: InferArena uses a single-step `schedule()` call, while Vidur's scheduler is hierarchical and event-driven. The adapter may need to buffer InferArena decisions into Vidur scheduler events.
- Model and GPU specs need to be added to `EngineSpec` before the adapter is useful.

## Recommendation

Keep the Vidur integration **conceptual for the MVP**. Add the adapter only after InferArena's plugin ecosystem and fast simulator are stable and a concrete use case (e.g., a paper submission) needs hardware-accurate validation.
