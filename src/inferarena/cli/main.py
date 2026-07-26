"""Command-line interface for InferArena."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from inferarena.core.experiment_spec import ExperimentSpec
from inferarena.core.plugin_registry import PluginRegistry
from inferarena.datasets.downloader import download_dataset, list_datasets
from inferarena.runner import ExperimentRunner

app = typer.Typer(help="InferArena: Experimentation platform for LLM inference systems")
console = Console()


DEFAULT_CONFIG = Path("experiment.yaml")


@app.command()
def run(
    config: Path = typer.Option(  # noqa: B008
        DEFAULT_CONFIG, help="Path to experiment config"
    ),
) -> None:
    """Run an experiment from a YAML config."""
    spec = ExperimentSpec.from_yaml(config)
    runner = ExperimentRunner()
    result = runner.run(spec)

    console.print(f"\n[bold]Scheduler:[/bold] {result.scheduler_name}")
    console.print(f"[bold]Completed:[/bold] {result.completed_requests}")

    table = Table(title="Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in result.summary().items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"\n[dim]Report saved to: {spec.output_dir}[/dim]")


@app.command()
def compare(
    config: Path = typer.Option(  # noqa: B008
        DEFAULT_CONFIG, help="Path to experiment config"
    ),
    schedulers: str = typer.Option("", help="Comma-separated list of schedulers to compare"),
) -> None:
    """Run the same workload with multiple schedulers and compare results."""
    base_spec = ExperimentSpec.from_yaml(config)
    registry = PluginRegistry()
    available = registry.list_schedulers()

    scheduler_names = [s.strip() for s in schedulers.split(",") if s.strip()]
    if not scheduler_names:
        scheduler_names = available

    invalid = [s for s in scheduler_names if s not in available]
    if invalid:
        raise typer.BadParameter(f"Unknown scheduler(s): {', '.join(invalid)}")

    runner = ExperimentRunner(registry)
    results = runner.compare(base_spec, scheduler_names)

    table = Table(title="Scheduler Comparison")
    headers = list(results[0].summary().keys()) if results else []
    for header in headers:
        table.add_column(header, style="cyan" if header == "scheduler" else "magenta")
    for result in results:
        row = [str(result.summary().get(h, "")) for h in headers]
        table.add_row(*row)
    console.print(table)
    console.print(f"\n[dim]Comparison report saved to: {base_spec.output_dir / 'comparison'}[/dim]")


@app.command()
def benchmark(
    directory: Path = typer.Option(  # noqa: B008
        Path("benchmarks"), help="Directory containing benchmark config YAML files"
    ),
) -> None:
    """Run all benchmark configs in a directory and aggregate results."""
    if not directory.exists():
        raise typer.BadParameter(f"Benchmark directory not found: {directory}")

    configs = sorted(directory.glob("*.yaml"))
    if not configs:
        raise typer.BadParameter(f"No benchmark configs found in {directory}")

    specs = [ExperimentSpec.from_yaml(path) for path in configs]
    runner = ExperimentRunner()
    results = runner.benchmark(specs)

    table = Table(title="Benchmark Summary")
    headers = list(results[0].summary().keys()) if results else []
    for header in headers:
        table.add_column(header, style="cyan" if header == "scheduler" else "magenta")
    for result in results:
        row = [str(result.summary().get(h, "")) for h in headers]
        table.add_row(*row)
    console.print(table)
    console.print("\n[dim]Benchmark summary saved to: inferarena_outputs/benchmarks/summary[/dim]")


@app.command("download-dataset")
def download_dataset_cmd(
    name_or_url: str = typer.Argument(help="Dataset alias or raw URL"),
    output: Path = typer.Option(  # noqa: B008
        Path("./datasets/sharegpt.json"), help="Output file path"
    ),
) -> None:
    """Download a public trace dataset."""
    path = download_dataset(name_or_url, output)
    console.print(f"[bold]Downloaded:[/bold] {path}")


@app.command("list-datasets")
def list_datasets_cmd() -> None:
    """List known downloadable datasets."""
    console.print("[bold]Available datasets:[/bold]")
    for name, url in list_datasets().items():
        console.print(f"  - {name}: {url}")


@app.command()
def list_schedulers() -> None:
    """List available schedulers."""
    registry = PluginRegistry()
    console.print("[bold]Available schedulers:[/bold]")
    for name in registry.list_schedulers():
        console.print(f"  - {name}")


@app.command("list-cache-policies")
def list_cache_policies_cmd() -> None:
    """List available cache policies."""
    registry = PluginRegistry()
    console.print("[bold]Available cache policies:[/bold]")
    for name in registry.list_cache_policies():
        console.print(f"  - {name}")


if __name__ == "__main__":
    app()
