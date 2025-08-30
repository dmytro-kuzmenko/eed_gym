# 01 — Install

## Prereqs
- Python 3.10+
- macOS/Linux/WSL

## Using uv (recommended)
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Using pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Verify
```bash
python -c "import eed_benchmark; print('OK')"
```
