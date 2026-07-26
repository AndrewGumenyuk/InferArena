# How to Add a Custom Router

Routers are inference components that decide which simulated worker handles an
incoming request in multi-GPU experiments.

1. Create a file in `src/inferarena/plugins/routers/my_router.py`.
2. Inherit from `inferarena.simulation.router.Router`.
3. Implement `route(self, request: Request, workers: list[Worker]) -> Worker`.
4. Register it in `src/inferarena/core/plugin_registry.py`.
5. Add tests.

## Example

```python
from inferarena.core.request import Request
from inferarena.simulation.router import Router
from inferarena.simulation.worker import Worker


class RandomRouter(Router):
    name = "random"

    def route(self, request: Request, workers: list[Worker]) -> Worker:
        import random

        return random.choice(workers)
```

Use the router in an experiment config:

```yaml
engine:
  name: multi_gpu_simulation
  num_workers: 4
  router: random
```
