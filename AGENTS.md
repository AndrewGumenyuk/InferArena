# Agent Instructions for InferArena

## Project Overview

InferArena is an open-source experimentation platform for LLM inference systems.
It lets researchers implement schedulers, cache policies, and routers as inference
components, and evaluate them fairly across simulation and real-cluster execution.

## Key Facts

- Language: Python 3.10+
- Build system: Hatch (pyproject.toml)
- Code style: Ruff + MyPy strict
- Docs framework: Diataxis
- License: Apache 2.0

## Common Commands

```bash
# Install dev environment
pip install -e ".[dev]"
pre-commit install

# Lint and format
ruff check .
ruff format .

# Type check
mypy src/inferarena

# Run tests
pytest

# Run a simple experiment
inferarena run --config examples/experiment.yaml
```

## Code Conventions

- Use `src/inferarena/` layout.
- All public functions and classes must be typed.
- Use Google-style docstrings.
- Inference components (schedulers, cache policies, routers) inherit from base classes in `inferarena.core`.
- Keep backend shims in `inferarena/engines/`.
- Do not use `print()` in library code; use `logging`.

## How to Add an Inference Component Plugin

### Scheduler

1. Create `src/inferarena/plugins/schedulers/my_scheduler.py`.
2. Inherit from `inferarena.core.scheduler.Scheduler`.
3. Implement `schedule()`.
4. Register it in `src/inferarena/core/plugin_registry.py`.
5. Add tests in `tests/unit/schedulers/test_my_scheduler.py`.
6. Add a how-to guide in `docs/how-to-guides/add-a-scheduler.md`.

### Cache Policy

1. Create `src/inferarena/plugins/cache_policies/my_cache.py`.
2. Inherit from `inferarena.core.cache_policy.CachePolicy`.
3. Implement `lookup()` and `store()`.
4. Register it in `src/inferarena/core/plugin_registry.py`.
5. Add tests in `tests/unit/test_my_cache.py`.
6. Add a how-to guide in `docs/how-to-guides/add-a-cache-policy.md`.

### Router

1. Create `src/inferarena/plugins/routers/my_router.py`.
2. Inherit from `inferarena.simulation.router.Router`.
3. Implement `route()`.
4. Register it in `src/inferarena/core/plugin_registry.py`.
5. Add tests in `tests/unit/test_my_router.py`.

## Architecture Decisions

- ADRs live in `architecture/decisions/`.
- Write an ADR for any decision affecting multiple components or quality attributes.

## What NOT to Do

- Do not commit code without tests unless it is purely documentation.
- Do not add dependencies without discussing in an issue first.
- Do not use `print()` in library code; use `logging`.
- Do not write docs that mix tutorial and reference styles.
