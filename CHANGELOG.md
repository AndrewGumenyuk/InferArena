# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-25

### Added
- Initial project scaffold with `pyproject.toml`, `src/` layout, and CI.
- Core abstractions: `Scheduler`, `Request`, `Batch`, `SystemState`, `TokenBudget`.
- Plugin registry with built-in scheduler discovery.
- Built-in schedulers: FCFS, ChunkedPrefill, Priority.
- Discrete-event simulation engine.
- Synthetic workload generation.
- Metrics collection and report generation (JSON, JSONL, Markdown).
- Typer-based CLI (`inferarena run`, `inferarena list-schedulers`).
- Tests and documentation skeleton.
