# Examples

* `dataset/` — a one-case synthetic dataset (96×64 reference, 13 frames, a 21-point array) and a synthetic
  prediction (64×44, twice the playback speed). It exercises video binding, automatic mappings and the MUSIQ worker
  without the benchmark download. Used by the GPU check in the [evaluator guide](../README.md#6-check-the-installation)
  and by `tests/test_cli_end_to_end.py`.
* `predictions.json`, `cases.json` — a prediction manifest and a case list for that dataset.
* [`rectified_submission/`](rectified_submission/README.md) — how to submit videos that were cropped, resized,
  undistorted or retimed, with `build_submission.py`.
* [`expected_outputs/`](expected_outputs/README.md) — the reports of the reference check on six benchmark cases
  (all 17 metrics), to compare an installation against.

From the repository root **[GPU]**:

```bash
sh evaluation/scripts/run_evaluation.sh evaluate \
  --dataset evaluation/examples/dataset \
  --predictions evaluation/examples/predictions.json --model Example \
  --metrics musiq --gpus 0 --foreground --run-dir runs/example
sh evaluation/scripts/run_evaluation.sh evaluate \
  --dataset evaluation/examples/dataset --cases evaluation/examples/cases.json \
  --use-reference-videos --metrics musiq --gpus 0 --foreground --run-dir runs/example-reference
```

Results: `runs/example/reports/scores.json`. `--videos evaluation/examples/dataset/predictions` works in place of
`--predictions` (binding by file name).
