# Published experiment results

Raw, versioned result artifacts from the experiments referenced in the docs.
Everything here is reproducible from the committed configs — see each
directory for the exact command.

## case-study/

Three-way scheduler comparison (FCFS vs SJF vs Sarathi-Serve) on the
variable-prompt-length workload, 64 requests, seed 42, 512-token step budget,
20,000-step limit.

Reproduce:

```bash
inferarena compare --config examples/case_study_variable.yaml \
  --schedulers fcfs,sjf,sarathi_serve
```

Analysis and interpretation: [docs/explanation/case-study.md](../docs/explanation/case-study.md).

## validation/

Cross-mode comparison of the same workload through the simulator and a mock
OpenAI-compatible server. This demonstrates the *semantic gap* between
simulation and API replay — it is **not** a calibration against real vLLM
(that experiment is scripted in `scripts/calibrate_against_vllm.py` and awaits
a GPU run).

Reproduce:

```bash
python scripts/mock_openai_server.py --port 8000 &
python scripts/validate_simulation.py
```

Analysis: [docs/explanation/validation.md](../docs/explanation/validation.md).
