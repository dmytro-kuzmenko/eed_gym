.PHONY: help venv install test lint docs-serve docs-build quickstart reproduce clean

help:
	@echo "Targets:"
	@echo "  venv         - create .venv (uv if available)"
	@echo "  install      - install package with dev extras in editable mode"
	@echo "  test         - run the test suite"
	@echo "  lint         - run ruff"
	@echo "  docs-serve   - serve MkDocs site locally"
	@echo "  docs-build   - build MkDocs site to site/"
	@echo "  quickstart   - run src/scripts/quickstart.sh"
	@echo "  reproduce    - train and evaluate every baseline and ablation"
	@echo "  clean        - remove build artifacts"

venv:
	@if command -v uv >/dev/null 2>&1; then uv venv .venv; else python -m venv .venv; fi
	@echo "Activate with: source .venv/bin/activate"

install:
	@if command -v uv >/dev/null 2>&1; then \
	  . .venv/bin/activate && uv pip install -e ".[dev,docs,analysis]"; \
	else \
	  . .venv/bin/activate && pip install -e ".[dev,docs,analysis]"; \
	fi

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check src tests

docs-serve:
	. .venv/bin/activate && mkdocs serve

docs-build:
	. .venv/bin/activate && mkdocs build

quickstart:
	bash src/scripts/quickstart.sh

reproduce:
	. .venv/bin/activate && bash src/scripts/reproduce.sh

clean:
	rm -rf site dist build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
