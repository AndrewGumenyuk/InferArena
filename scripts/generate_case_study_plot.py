#!/usr/bin/env python3
"""Generate a comparison plot for the variable-prompt case study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    base = Path("inferarena_outputs/case_study_variable/comparison/comparison.json")
    if not base.exists():
        raise FileNotFoundError(base)

    data = json.loads(base.read_text())
    schedulers = [d["scheduler"] for d in data]
    completed = [d["completed_requests"] for d in data]
    throughput = [d["throughput_rps"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(schedulers, completed, color=["#e74c3c", "#2ecc71"])
    axes[0].set_ylabel("Completed requests")
    axes[0].set_title("Completed requests (64 total)")
    axes[0].set_ylim(0, max(completed) * 1.2)
    for i, v in enumerate(completed):
        axes[0].text(i, v + 1, str(v), ha="center")

    axes[1].bar(schedulers, throughput, color=["#e74c3c", "#2ecc71"])
    axes[1].set_ylabel("Throughput (req/s)")
    axes[1].set_title("Throughput under 20,000-step budget")
    axes[1].set_ylim(0, max(throughput) * 1.2)
    for i, v in enumerate(throughput):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.suptitle("FCFS vs SJF on variable prompt lengths")
    plt.tight_layout()

    output = Path("docs/assets/case-study-comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
