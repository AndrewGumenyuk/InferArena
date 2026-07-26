"""Report generator for InferArena experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferarena.metrics.collector import MetricsCollector
from inferarena.metrics.result import ExperimentResult
from inferarena.reporting import plots


class ReportGenerator:
    """Generates reproducible reports from experiment results."""

    def __init__(self, result: ExperimentResult, metrics: MetricsCollector) -> None:
        """Initialize the report generator."""
        self.result = result
        self.metrics = metrics

    @classmethod
    def from_result(cls, result: ExperimentResult, metrics: MetricsCollector) -> ReportGenerator:
        """Create a report generator from a result and metrics collector."""
        return cls(result, metrics)

    def save(self, output_dir: Path | str) -> None:
        """Save the report to the given output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.result.summary()
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        telemetry_path = output_dir / "telemetry.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as f:
            for record in self.metrics.to_records():
                f.write(json.dumps(record) + "\n")

        requests_path = output_dir / "requests.json"
        with requests_path.open("w", encoding="utf-8") as f:
            json.dump([self._request_to_dict(r) for r in self.result.request_results], f, indent=2)

        markdown_path = output_dir / "report.md"
        with markdown_path.open("w", encoding="utf-8") as f:
            f.write(self._to_markdown())

        self._save_plots(output_dir)

    def _save_plots(self, output_dir: Path) -> None:
        """Generate plots if matplotlib is installed."""
        try:
            plots.plot_telemetry(self.metrics.to_records(), output_dir / "telemetry.png")
            latencies = [
                r.e2e_latency for r in self.result.request_results if r.e2e_latency is not None
            ]
            if latencies:
                plots.plot_latency_cdf(latencies, "E2E Latency", output_dir / "latency_cdf.png")
        except ImportError:
            # Plotting is optional; skip if matplotlib is missing.
            pass

    def _to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# InferArena Experiment Report",
            "",
            f"**Scheduler:** {self.result.scheduler_name}",
            f"**Completed Requests:** {self.result.completed_requests}",
            f"**Total Steps:** {self.result.total_steps}",
            f"**Total Time:** {self.result.total_time:.2f} ms",
            f"**Throughput:** {self.result.throughput:.2f} req/s",
            "",
            "## Summary Metrics",
            "",
        ]
        for key, value in self.result.summary().items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a dictionary."""
        return {
            "summary": self.result.summary(),
            "telemetry": self.metrics.to_records(),
        }

    @staticmethod
    def _request_to_dict(request: Any) -> dict[str, Any]:
        """Serialize a RequestResult to a plain dictionary."""
        return {
            "request_id": request.request_id,
            "arrival_time": request.arrival_time,
            "prompt_tokens": request.prompt_tokens,
            "max_output_tokens": request.max_output_tokens,
            "scheduled_time": request.scheduled_time,
            "first_token_time": request.first_token_time,
            "completion_time": request.completion_time,
            "queue_time": request.queue_time,
            "prefill_duration": request.prefill_duration,
            "ttft": request.ttft,
            "tbt": request.tbt,
            "e2e_latency": request.e2e_latency,
        }
