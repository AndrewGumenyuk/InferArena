"""Downloader for public LLM-serving trace datasets."""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Known public dataset aliases. URLs may change; verify before relying on them.
DATASET_ALIASES: dict[str, str] = {
    "sharegpt_vicuna": (
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
        "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    ),
}


def resolve_url(name_or_url: str) -> str:
    """Resolve a dataset alias or pass through a raw URL."""
    return DATASET_ALIASES.get(name_or_url, name_or_url)


def download_dataset(name_or_url: str, output_path: Path | str) -> Path:
    """Download a dataset to the given output path.

    Args:
        name_or_url: Either a known alias (e.g. ``sharegpt_vicuna``) or a raw URL.
        output_path: Local path to write the downloaded file.

    Returns:
        The path to the downloaded file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_url(name_or_url)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "InferArena dataset downloader"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
        data = response.read()
        output_path.write_bytes(data)

    return output_path


def list_datasets() -> dict[str, str]:
    """Return the mapping of known dataset aliases to URLs."""
    return dict(DATASET_ALIASES)
