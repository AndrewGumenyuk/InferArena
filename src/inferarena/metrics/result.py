"""Experiment result and metric models."""

from __future__ import annotations

from dataclasses import dataclass, field


def _percentile(values: list[float], p: float) -> float:
    """Return the percentile of a sorted list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


@dataclass
class RequestResult:
    """Metrics for a single request."""

    request_id: str
    arrival_time: float
    prompt_tokens: int
    max_output_tokens: int
    scheduled_time: float | None = None
    first_token_time: float | None = None
    completion_time: float | None = None

    @property
    def queue_time(self) -> float | None:
        """Time spent waiting before the first prefill step (ms)."""
        if self.scheduled_time is None:
            return None
        return self.scheduled_time - self.arrival_time

    @property
    def prefill_duration(self) -> float | None:
        """Time spent in the prefill phase (ms)."""
        if self.scheduled_time is None or self.first_token_time is None:
            return None
        return self.first_token_time - self.scheduled_time

    @property
    def ttft(self) -> float | None:
        """Time to first token (ms)."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def tbt(self) -> float | None:
        """Average time between tokens during decode (ms)."""
        if self.first_token_time is None or self.completion_time is None:
            return None
        generated = self.completion_time - self.first_token_time
        tokens = self.max_output_tokens
        if tokens <= 0:
            return None
        return generated / tokens

    @property
    def e2e_latency(self) -> float | None:
        """End-to-end latency (ms)."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


@dataclass
class ExperimentResult:
    """Aggregated result of an experiment."""

    scheduler_name: str
    total_steps: int
    total_time: float
    completed_requests: int
    request_results: list[RequestResult] = field(default_factory=list)
    cache_hits: int = 0
    cache_lookups: int = 0

    @property
    def throughput(self) -> float:
        """Requests per second."""
        if self.total_time <= 0:
            return 0.0
        return self.completed_requests / (self.total_time / 1000.0)

    def summary(self) -> dict[str, float | int | str]:
        """Return a dictionary summary of key metrics."""
        ttfts = [r.ttft for r in self.request_results if r.ttft is not None]
        latencies = [r.e2e_latency for r in self.request_results if r.e2e_latency is not None]
        queue_times = [r.queue_time for r in self.request_results if r.queue_time is not None]
        tbts = [r.tbt for r in self.request_results if r.tbt is not None]
        prefill_durations = [
            r.prefill_duration for r in self.request_results if r.prefill_duration is not None
        ]
        return {
            "scheduler": self.scheduler_name,
            "completed_requests": self.completed_requests,
            "total_steps": self.total_steps,
            "total_time_ms": round(self.total_time, 2),
            "throughput_rps": round(self.throughput, 2),
            "ttft_p50_ms": round(_percentile(ttfts, 0.5), 2) if ttfts else 0.0,
            "ttft_p99_ms": round(_percentile(ttfts, 0.99), 2) if ttfts else 0.0,
            "latency_p50_ms": round(_percentile(latencies, 0.5), 2) if latencies else 0.0,
            "latency_p99_ms": round(_percentile(latencies, 0.99), 2) if latencies else 0.0,
            "queue_time_p50_ms": round(_percentile(queue_times, 0.5), 2) if queue_times else 0.0,
            "tbt_p50_ms": round(_percentile(tbts, 0.5), 2) if tbts else 0.0,
            "prefill_p50_ms": round(_percentile(prefill_durations, 0.5), 2)
            if prefill_durations
            else 0.0,
            "cache_hits": self.cache_hits,
            "cache_lookups": self.cache_lookups,
            "cache_hit_rate": round(self.cache_hits / self.cache_lookups, 4)
            if self.cache_lookups
            else 0.0,
        }
