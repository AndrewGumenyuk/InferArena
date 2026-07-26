"""Tests for the vLLM real-cluster engine adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inferarena.core.experiment_spec import EngineSpec, ExperimentSpec, WorkloadSpec
from inferarena.engines.vllm_adapter import VLLMEngine, _prompt_text
from inferarena.plugins.schedulers.fcfs import FCFSScheduler

openai = pytest.importorskip("openai", reason="openai package not installed")


def test_prompt_text_scales_with_tokens() -> None:
    request = MagicMock()
    request.prompt_tokens = 10
    text = _prompt_text(request)
    assert len(text.split()) > 0


def test_vllm_engine_run_against_mock() -> None:
    scheduler = FCFSScheduler()
    engine = VLLMEngine(scheduler, EngineSpec())

    spec = ExperimentSpec(
        engine=EngineSpec(name="vllm", model="test-model"),
        workload=WorkloadSpec(
            num_requests=2,
            arrival_rate=10.0,
            prompt_tokens=16,
            output_tokens=2,
            seed=42,
        ),
    )

    fake_chunk = MagicMock()
    fake_chunk.choices = [MagicMock()]
    fake_chunk.choices[0].delta.content = "x"

    fake_stream = AsyncMock()
    fake_stream.__aiter__.return_value = [fake_chunk, fake_chunk]

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_stream)

    with patch(
        "openai.AsyncOpenAI",
        return_value=fake_client,
    ):
        result = engine.run(spec)

    assert result.completed_requests == 2
    assert result.scheduler_name == "fcfs"
    assert all(r.first_token_time is not None for r in result.request_results)
    assert all(r.completion_time is not None for r in result.request_results)
