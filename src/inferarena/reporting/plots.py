"""Telemetry plotting utilities for InferArena reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _get_matplotlib() -> Any:
    """Import matplotlib lazily so plotting remains optional."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install inferarena[plot]"
        ) from exc


def plot_telemetry(telemetry: list[dict[str, Any]], output_path: Path | str) -> None:
    """Generate a multi-panel telemetry figure and save it.

    Args:
        telemetry: List of per-step metric records.
        output_path: Path to write the PNG figure.
    """
    plt = _get_matplotlib()
    output_path = Path(output_path)

    times = [r["time"] for r in telemetry]
    batch_sizes = [r["batch_size"] for r in telemetry]
    batch_tokens = [r["batch_tokens"] for r in telemetry]
    waiting = [r["waiting_count"] for r in telemetry]
    running = [r["running_count"] for r in telemetry]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(times, batch_sizes, label="Batch size")
    axes[0, 0].set_title("Batch Size Over Time")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Requests")

    axes[0, 1].plot(times, batch_tokens, color="orange")
    axes[0, 1].set_title("Batch Tokens Over Time")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Tokens")

    axes[1, 0].plot(times, waiting, label="Waiting")
    axes[1, 0].plot(times, running, label="Running")
    axes[1, 0].set_title("Queue State Over Time")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Requests")
    axes[1, 0].legend()

    if batch_sizes:
        axes[1, 1].hist(batch_sizes, bins=max(1, len(set(batch_sizes))), color="green", alpha=0.7)
    axes[1, 1].set_title("Batch Size Distribution")
    axes[1, 1].set_xlabel("Requests per step")
    axes[1, 1].set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latency_cdf(values: list[float], label: str, output_path: Path | str) -> None:
    """Plot a CDF of latency values."""
    plt = _get_matplotlib()
    output_path = Path(output_path)

    sorted_values = sorted(values)
    y = [i / len(sorted_values) for i in range(1, len(sorted_values) + 1)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_values, y, label=label)
    ax.set_title(f"{label} CDF")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
