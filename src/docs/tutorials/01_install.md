# 01 — Install

```bash
git clone https://github.com/dmytro-kuzmenko/eed_gym
cd eed_gym
uv venv .venv && source .venv/bin/activate
uv pip install -e .          # or: pip install -e .
```

Extras: `.[dev]` for tests and linting, `.[docs]` for this site, `.[analysis]` for the vignette statistics script.

Check the install:

```bash
python -c "import gymnasium as gym, eed_benchmark; print(gym.make(eed_benchmark.ENV_ID).reset(seed=0)[0])"
```
