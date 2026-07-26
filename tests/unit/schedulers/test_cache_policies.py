"""Tests for cache policies."""

from inferarena.core.request import Request
from inferarena.plugins.cache_policies.no_op import NoOpCachePolicy
from inferarena.plugins.cache_policies.prefix import PrefixCache


def test_no_op_cache_returns_zero() -> None:
    policy = NoOpCachePolicy()
    request = Request(arrival_time=0.0, prompt_tokens=100, max_output_tokens=5)
    assert policy.lookup(request) == 0


def test_prefix_cache_repeated_prompt() -> None:
    policy = PrefixCache(max_prefixes=10)
    request = Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5)
    assert policy.lookup(request) == 0
    policy.store(request)

    # Exact same prompt should be fully cached.
    repeat = Request(arrival_time=1.0, prompt_tokens=10, max_output_tokens=5)
    assert policy.lookup(repeat) == 10


def test_prefix_cache_longer_prompt_reuses_prefix() -> None:
    policy = PrefixCache(max_prefixes=10)
    first = Request(arrival_time=0.0, prompt_tokens=10, max_output_tokens=5)
    policy.store(first)

    longer = Request(arrival_time=1.0, prompt_tokens=15, max_output_tokens=5)
    assert policy.lookup(longer) == 10


def test_prefix_cache_evicts_oldest() -> None:
    policy = PrefixCache(max_prefixes=2)
    a = Request(arrival_time=0.0, prompt_tokens=5, max_output_tokens=5)
    b = Request(arrival_time=1.0, prompt_tokens=6, max_output_tokens=5)
    c = Request(arrival_time=2.0, prompt_tokens=7, max_output_tokens=5)
    policy.store(a)
    policy.store(b)
    policy.store(c)

    assert policy.lookup(a) == 0
    assert policy.lookup(b) == 6
    assert policy.lookup(c) == 7
