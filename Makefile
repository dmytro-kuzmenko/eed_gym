.PHONY: help venv install docs-serve docs-build quickstart clean

help:
	@echo "Targets:"
	@echo "  venv         - create .venv (uv if available)"
	@echo "  install      - install package in editable mode"
	@echo "  docs-serve   - serve MkDocs site locally"
	@echo "  docs-build   - build MkDocs site to site/"
	@echo "  quickstart   - run quickstart.sh"
	@echo "  clean        - remove build artifacts"

venv:
	@if command -v uv >/dev/null 2>&1; then \
	  uv venv .venv; \
	else \
	  python -m venv .venv; \
	fi
	@echo "Activate with: source .venv/bin/activate"

install:
	@if command -v uv >/dev/null 2>&1; then \
	  . .venv/bin/activate && uv pip install -e .; \
	else \
	  . .venv/bin/activate && pip install -e .; \
	fi

docs-serve:
	. .venv/bin/activate && pip install -q mkdocs mkdocs-material "mkdocstrings[python]" && mkdocs serve

docs-build:
	. .venv/bin/activate && pip install -q mkdocs mkdocs-material "mkdocstrings[python]" && mkdocs build

quickstart:
	bash quickstart.sh

clean:
	rm -rf site dist build __pycache__ .pytest_cache
