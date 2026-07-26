"""Streamlit dashboard for InferArena experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None  # type: ignore[assignment]

from inferarena.reporting.dashboard import discover_experiments, load_request_results


def _summaries_to_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert summary dictionaries to a DataFrame."""
    return pd.DataFrame(summaries)


def _plot_latency_cdf(summaries: list[dict[str, Any]]) -> Any:
    """Plot latency CDFs across selected experiments."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
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
    return fig


def _require_streamlit() -> None:
    """Raise a helpful error if streamlit is not installed."""
    if st is None:
        raise ImportError(
            "Dashboard requires streamlit. Install it with: pip install inferarena[dashboard]"
        )


def main() -> None:
    """Run the InferArena Streamlit dashboard."""
    _require_streamlit()
    st.set_page_config(page_title="InferArena Dashboard", layout="wide")
    st.title("InferArena Dashboard")
    st.markdown("Explore experiment results produced by the `inferarena` CLI.")

    output_dir = st.sidebar.text_input("Output directory", value="inferarena_outputs")
    summaries = discover_experiments(output_dir)

    if not summaries:
        st.warning(f"No experiment summaries found in `{output_dir}`.")
        st.info(
            "Run an experiment first, e.g. `inferarena compare --config examples/experiment.yaml`."
        )
        return

    st.sidebar.metric("Experiments discovered", len(summaries))

    df = _summaries_to_df(summaries)

    st.header("Summary table")
    summary_cols = [
        "scheduler",
        "completed_requests",
        "throughput_rps",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "latency_p50_ms",
        "latency_p99_ms",
        "cache_hit_rate",
    ]
    display_cols = [c for c in summary_cols if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values("throughput_rps", ascending=False), use_container_width=True
    )

    st.header("Latency CDF")
    selected = st.multiselect(
        "Select experiments",
        options=[s.get("scheduler", s.get("path", "unknown")) for s in summaries],
        default=[s.get("scheduler", "unknown") for s in summaries],
    )
    selected_summaries = [
        s for s in summaries if s.get("scheduler", s.get("path", "unknown")) in selected
    ]
    if selected_summaries:
        fig = _plot_latency_cdf(selected_summaries)
        st.pyplot(fig)

    st.header("Raw summaries")
    st.json(summaries)


def run_dashboard(output_dir: Path | str = "inferarena_outputs") -> None:
    """Launch the Streamlit dashboard (entry point for CLI and tests)."""
    _require_streamlit()
    import sys

    import streamlit.web.cli as stcli

    app_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", app_path, "--", str(output_dir)]
    stcli.main()


if __name__ == "__main__":
    main()
