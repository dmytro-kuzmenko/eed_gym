# Contributing

Thanks for your interest in EED-Gym!

- Use feature branches and PRs.
- Keep functions small and documented (Google-style docstrings).
- Add a test for every behaviour change. CI runs `ruff` and `pytest` on each PR.
- Changes to env dynamics or metric definitions need an entry in `CHANGELOG.md`, and in `ERRATA.md` if they affect published numbers.

## Dev setup
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
ruff check src tests
```
