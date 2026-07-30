# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-27

First public release.

### Added

- **Plugin architecture** for inference components: schedulers, cache
  policies, and routers implement small interfaces while the framework handles
  workloads, execution, metrics, and reporting.
- **Discrete-event simulation engine** with a token-budget step model and
  chunked prefill support (`src/inferarena/simulation/engine.py`).
- **Built-in schedulers**: FCFS, SJF, Priority, Round-Robin, Chunked-Prefill,
  and a faithful reproduction of **Sarathi-Serve** (arXiv:2403.02310,
  Algorithm 3).
- **Real-engine adapters** (API replay mode) for vLLM, SGLang, and
  TensorRT-LLM via the OpenAI-compatible API.
- **Workloads**: seeded synthetic generators (uniform, variable) and ShareGPT
  trace replay.
- **CLI**: `inferarena run`, `inferarena compare`, `inferarena list-schedulers`,
  and more, with YAML experiment configs.
- **Metrics and reporting**: per-request lifecycle timestamps, per-step
  telemetry, percentiles, comparison tables, Markdown reports, and plots.
- **Case study**: FCFS vs SJF vs Sarathi-Serve under variable prompt lengths
  (2/64 vs 33/64 vs 64/64 completions), with honest treatment of survivorship
  bias and starvation (`docs/explanation/case-study.md`, raw artifacts in
  `results/`).
- **Fidelity documentation**: simulation assumptions and limitations
  (`docs/explanation/simulation-assumptions.md`) and the three-mode execution
  taxonomy with validation status (`docs/explanation/validation.md`).
- **Calibration tooling**: `scripts/calibrate_against_vllm.py` measures real
  per-token costs on a live vLLM deployment and compares calibrated simulation
  against it (guide: `docs/how-to-guides/calibrate-against-a-real-engine.md`).
- Web dashboard (optional Streamlit extra), Jupyter dashboard, Docker demo
  with a mock OpenAI server, and full CI (Ruff, MyPy, Pytest).

### Known limitations

- Scheduler plugins run in simulation only; real-engine adapters replay
  workloads and measure latency from the outside (native engine integration is
  planned).
- The simulator has no KV-cache memory model, no preemption cost, and a flat
  decode-time model — see the simulation assumptions doc before trusting
  absolute millisecond numbers.
- The simulator has not yet been calibrated against GPU hardware; the tooling
  and procedure are published and results are welcome.
