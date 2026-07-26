# How to Add a Custom Cache Policy

Cache policies decide how much of a request's prompt is already cached from
previous requests. The simulation engine uses this to reduce prefill cost.

## Example

```python
from inferarena import CachePolicy, Request


class MyCachePolicy(CachePolicy):
    name = "my_cache"

    def lookup(self, request: Request) -> int:
        # Return the number of cached prompt tokens.
        return 0

    def store(self, request: Request) -> None:
        # Remember the request's prompt after prefill.
        pass
```

## Register the policy

Add your class to `src/inferarena/core/plugin_registry.py`:

```python
from my_package import MyCachePolicy

self.register_cache_policy(MyCachePolicy)
```

## Use it in an experiment

Set `cache_policy: my_cache` in your YAML config:

```yaml
name: "my-experiment"
scheduler: "fcfs"
cache_policy: "my_cache"
workload:
  name: "uniform"
  num_requests: 32
  prompt_tokens: 512
  output_tokens: 128
```

## Metrics

The experiment report includes `cache_hits`, `cache_lookups`, and
`cache_hit_rate` when a cache policy is active.
