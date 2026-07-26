# Getting Started

## Installation

```bash
git clone https://github.com/yourorg/inferarena.git
cd inferarena
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Your First Experiment

```bash
inferarena run --config examples/experiment.yaml
```

## List Available Schedulers

```bash
inferarena list-schedulers
```
