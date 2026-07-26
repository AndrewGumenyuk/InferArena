"""Tests for the dataset downloader."""

import json

from inferarena.datasets.downloader import DATASET_ALIASES, download_dataset, resolve_url


def test_resolve_url_returns_alias() -> None:
    url = resolve_url("sharegpt_vicuna")
    assert url == DATASET_ALIASES["sharegpt_vicuna"]


def test_resolve_url_passes_through_raw_url() -> None:
    raw = "https://example.com/dataset.json"
    assert resolve_url(raw) == raw


def test_download_dataset_writes_file(tmp_path) -> None:
    # Use a tiny JSON file served by httpbin as a stable test target.
    sample = {"conversations": [{"from": "human", "value": "hi"}]}
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")

    output = tmp_path / "downloaded.json"
    result = download_dataset(f"file://{sample_path}", output)
    assert result.exists()
    assert result.read_text(encoding="utf-8") == sample_path.read_text(encoding="utf-8")
