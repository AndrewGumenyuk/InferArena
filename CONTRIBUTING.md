# Contributing to InferArena

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/yourorg/inferarena.git
cd inferarena
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Run Checks

```bash
ruff check .
ruff format .
mypy src/inferarena
pytest
```

## How to Contribute

1. Open or find an issue.
2. Create a feature branch.
3. Make changes with tests and docs.
4. Ensure CI passes.
5. Open a pull request.

## License

By contributing, you agree that your contributions will be licensed under Apache License 2.0.
