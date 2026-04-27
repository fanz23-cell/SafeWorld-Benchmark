# SafeWorld-Benchmark

This repository contains a first implementation of the SAFEWORLD-BENCH benchmark task layer for Safety Gymnasium.

Current contents:

- `benchmark/`: task configs, AP extractors, evaluators, runners
- `scripts/`: inspection and benchmark run scripts
- `README_benchmark_tasks.md`: detailed benchmark-task documentation
- `run_safety_envs.py`, `show_safety_envs.py`: local environment inspection and visualization helpers

Notes:

- This is the benchmark task layer only.
- Generated artifacts are intentionally ignored through `.gitignore`.
- See `README_benchmark_tasks.md` for task definitions, grounding notes, and usage.
