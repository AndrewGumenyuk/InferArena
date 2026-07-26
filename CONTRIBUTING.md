# Contributing to InferArena

Thank you for your interest in contributing! InferArena is a young project, and thoughtful contributions make a big difference.

## Development setup

```bash
git clone https://github.com/AndrewGumenyuk/InferArena.git
cd InferArena
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"
pre-commit install
```

## Run checks before committing

```bash
ruff check .
ruff format --check .
mypy src/inferarena
pytest
```

Auto-fix formatting and import order:

```bash
ruff format .
ruff check . --fix
```

## How to contribute

1. **Open an issue** to discuss your idea or claim an existing one.
2. **Fork the repository** and create a feature branch.
3. **Make your changes** with tests and documentation.
4. **Ensure CI passes** locally before opening a PR.
5. **Open a pull request** and fill out the template.

## What makes a great contribution

- **Small, focused changes** are easier to review than large refactors.
- **Tests are required** for new behavior. Pure documentation changes are the exception.
- **Docs keep the project usable.** Update README, how-to guides, or architecture notes when behavior changes.
- **Follow existing conventions.** See `AGENTS.md` for the project style guide.

## Ideas for first contributions

- Add a new scheduler, cache policy, or router.
- Improve the simulation model (e.g., more accurate prefill/decode costs).
- Add an example config for a public dataset.
- Improve the Streamlit or Jupyter dashboard.
- Write additional how-to guides.

## License

By contributing, you agree that your contributions will be licensed under Apache License 2.0.
