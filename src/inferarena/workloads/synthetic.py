"""Synthetic workload generators."""

from __future__ import annotations

import random

from inferarena.core.request import Request


def generate_uniform(
    num_requests: int,
    arrival_rate: float,
    prompt_tokens: int,
    max_output_tokens: int,
    seed: int = 42,
) -> list[Request]:
    """Generate a synthetic workload with uniform arrival and fixed lengths.

    Args:
        num_requests: Number of requests to generate.
        arrival_rate: Average number of requests per second.
        prompt_tokens: Fixed prompt length for all requests.
        max_output_tokens: Fixed max output length for all requests.
        seed: Random seed for reproducibility.

    Returns:
        A list of Request objects sorted by arrival time.
    """
    rng = random.Random(seed)
    requests: list[Request] = []
    current_time = 0.0
    for _ in range(num_requests):
        inter_arrival = rng.expovariate(arrival_rate)
        current_time += inter_arrival
        requests.append(
            Request(
                arrival_time=current_time * 1000.0,  # convert to ms
                prompt_tokens=prompt_tokens,
                max_output_tokens=max_output_tokens,
            )
        )
    return requests


def generate_variable(
    num_requests: int,
    arrival_rate: float,
    prompt_tokens_range: tuple[int, int],
    output_tokens_range: tuple[int, int],
    seed: int = 42,
) -> list[Request]:
    """Generate a synthetic workload with variable prompt/output lengths.

    Args:
        num_requests: Number of requests to generate.
        arrival_rate: Average number of requests per second.
        prompt_tokens_range: (min, max) prompt length.
        output_tokens_range: (min, max) output length.
        seed: Random seed for reproducibility.

    Returns:
        A list of Request objects sorted by arrival time.
    """
    rng = random.Random(seed)
    requests: list[Request] = []
    current_time = 0.0
    for _ in range(num_requests):
        inter_arrival = rng.expovariate(arrival_rate)
        current_time += inter_arrival
        prompt_tokens = rng.randint(*prompt_tokens_range)
        max_output_tokens = rng.randint(*output_tokens_range)
        requests.append(
            Request(
                arrival_time=current_time * 1000.0,
                prompt_tokens=prompt_tokens,
                max_output_tokens=max_output_tokens,
            )
        )
    return requests
