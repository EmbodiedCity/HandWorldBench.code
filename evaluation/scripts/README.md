# Scripts

* `run_evaluation.sh` — the launcher used in the [evaluator guide](../README.md). It finds `.envs/main` in the
  repository root, sets `PYTHONPATH` to this checkout and, for a new run, defaults `--dataset` to
  `<repository root>/HandWorldBench-data` and `--config` to `configs/runtime.json`. All other arguments go to
  `python -m handworldbench`.
* `build_runtime_config.py` — writes `configs/runtime.json` from `configs/runtime.example.json` for the installed
  environments and weights (`--profile release --release-root .`, or `--profile custom` with explicit paths;
  `--check` verifies everything exists). See step 5 of the guide.
