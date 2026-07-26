"""Programmatic helpers for the Jupyter notebook dashboard.

The dashboard loads experiment results from ``inferarena_outputs`` and produces
comparison tables and latency CDF plots across schedulers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def discover_experiments(root: Path | str = "inferarena_outputs") -> list[dict[str, Any]]:
    """Discover all ``summary.json`` files under *root* and load them.

    Args:
        root: Directory to search for experiment outputs.

    Returns:
        List of summary dictionaries with an added ``path`` key.
    """
    root = Path(root)
    summaries: list[dict[str, Any]] = []
    if not root.exists():
        return summaries
    for summary_path in root.rglob("summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["path"] = str(summary_path.parent)
            summaries.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def load_request_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Load per-request results if they were serialized alongside the summary.

    Args:
        summary: Summary dictionary returned by :func:`discover_experiments`.

    Returns:
        List of request result dictionaries, or an empty list if none found.
    """
    request_path = Path(summary["path"]) / "requests.json"
    if not request_path.exists():
        return []
    try:
        with request_path.open("r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
            return data
    except (json.JSONDecodeError, OSError):
        return []


def to_dataframe(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of summary dictionaries to a pandas DataFrame."""
    return pd.DataFrame(summaries)


def plot_latency_cdf(summaries: list[dict[str, Any]], ax: Any | None = None) -> Any:
    """Plot end-to-end latency CDFs for each experiment summary.

    Args:
        summaries: Summary dictionaries with loaded request results.
        ax: Optional matplotlib axes to draw on.

    Returns:
        The matplotlib axes used.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for summary in summaries:
        requests = load_request_results(summary)
        latencies = [
            r["completion_time"] - r["arrival_time"]
            for r in requests
            if r.get("completion_time") is not None and r.get("arrival_time") is not None
        ]
        if not latencies:
            continue
        sorted_latencies = sorted(latencies)
        cdf = [(i + 1) / len(sorted_latencies) for i in range(len(sorted_latencies))]
        label = summary.get("scheduler", "unknown")
        ax.plot(sorted_latencies, cdf, label=label, marker=".", linestyle="-")

    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF by scheduler")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax
