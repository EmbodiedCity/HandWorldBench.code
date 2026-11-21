# Runtime configuration

* `runtime.json` — the configuration the launcher reads by default. It is not part of the repository: generate it
  for your machine with `scripts/build_runtime_config.py` (step 5 of the
  [evaluator guide](../README.md#5-bind-the-runtime-configuration)). Paths are relative to this directory and assume
  the repository layout of the guide (`.envs/`, `models/`, `cache/` in the repository root).
* `runtime.example.json` — the template with `${MODELS_ROOT}`-style placeholders.

Sampling (16 frames), batch sizes and backend separation in these files are part of the benchmark protocol.
