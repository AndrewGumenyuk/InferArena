# syntax=docker/dockerfile:1

# InferArena experimentation platform
# Usage:
#   docker build -t inferarena .
#   docker run --rm -it -v $(pwd)/inferarena_outputs:/app/inferarena_outputs inferarena run --config examples/experiment.yaml

FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies (needed for compiled extras like vLLM/SGLang if installed).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for installation first to leverage layer caching.
COPY pyproject.toml README.md ./
COPY src/inferarena/__init__.py src/inferarena/__init__.py

# Install the package in editable mode with plotting support.
# Use [all] instead of [plot] if you need real-cluster engine adapters.
RUN pip install --upgrade pip && \
    pip install -e ".[plot]"

# Copy the rest of the source code.
COPY src/ src/
COPY examples/ examples/
COPY benchmarks/ benchmarks/

# Default entrypoint exposes the InferArena CLI.
ENTRYPOINT ["inferarena"]
CMD ["--help"]

# Development target with linting and testing tools.
FROM base AS dev

RUN pip install -e ".[dev,plot]"

COPY tests/ tests/
COPY docs/ docs/
COPY scripts/ scripts/

# Reset entrypoint so the default command runs pytest directly.
ENTRYPOINT []
CMD ["pytest"]
