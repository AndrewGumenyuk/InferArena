"""Comparison reports across multiple schedulers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferarena.metrics.result import ExperimentResult


class ComparisonReportGenerator:
    """Generate side-by-side comparison reports for multiple schedulers."""

    def __init__(self, results: list[ExperimentResult]) -> None:
        """Initialize with results from multiple schedulers."""
        self.results = results

    def to_table(self) -> list[dict[str, Any]]:
        """Return a list of summary dictionaries, one per scheduler."""
        return [result.summary() for result in self.results]

    def to_markdown(self) -> str:
        """Generate a markdown comparison table."""
        rows = self.to_table()
        if not rows:
            return "# Scheduler Comparison\n\nNo results to compare.\n"

        headers = list(rows[0].keys())
        lines = ["# Scheduler Comparison", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def save(self, output_dir: Path | str) -> None:
        """Save comparison report to the given directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "comparison.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_table(), f, indent=2)

        markdown_path = output_dir / "comparison.md"
        with markdown_path.open("w", encoding="utf-8") as f:
            f.write(self.to_markdown())
