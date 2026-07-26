# How to Add a Custom Scheduler

1. Create a file in `src/inferarena/plugins/schedulers/my_scheduler.py`.
2. Inherit from `inferarena.core.scheduler.Scheduler`.
3. Implement `schedule(self, state: SystemState) -> Batch`.
4. Register it via entry points or in `src/inferarena/core/plugin_registry.py`.
5. Add tests.

## Example

```python
from inferarena.core.batch import Batch
from inferarena.core.scheduler import Scheduler
from inferarena.core.system_state import SystemState


class GreedyScheduler(Scheduler):
    name = "greedy"

    def schedule(self, state: SystemState) -> Batch:
        batch = Batch()
        used = 0
        for request in state.running + state.waiting:
            cost = 1 if request.is_prefill_complete else request.prompt_tokens
            if used + cost <= state.budget.max_tokens:
                batch.requests.append(request)
                used += cost
        return batch
```
