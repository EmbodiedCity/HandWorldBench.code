# Expected outputs: reference check on six cases

These are the reports of the reference check in [§6 of the evaluator guide](../../README.md#6-check-the-installation):
the reference videos of six benchmark cases (three human, one HandsOn, two TI5; together they cover every metric
family, including Memory Consistency, Event Adherence and State Regulation / Progression) scored with all 17 metrics.
Run the same command and compare your `runs/reference6/reports/` with the files here.

```bash
echo '["hw500_0014", "hw500_0363", "hw500_0431", "hw500_0050", "hw500_0150", "hw500_0452"]' > cases6.json
sh evaluation/scripts/run_evaluation.sh evaluate --use-reference-videos \
  --cases cases6.json --gpus 0,1 --foreground --run-dir runs/reference6
```

Recorded on two NVIDIA H20 GPUs with the environments and weights of §2–§5 and without `causal-conv1d`
(201 tasks, about 31 min including model loading).

| File | Content |
| --- | --- |
| `scores.json` | one record per (case, metric): 102 records with the per-metric details the evaluator writes |
| `summary.json` | per-metric means over the common cases, overall and per embodiment |
| `hwbscore.json` | six dimension means and the 0–100 HWBScore, overall and per embodiment |

Paths inside the reports are relative to the repository root (`HandWorldBench-data/...`, `cache/...`, `models/...`,
`runs/reference6/...`); the cache files they name are not shipped. Task ids and cache keys are hashes of the task
inputs and of the run setup, so compare scores and statuses, not ids.

## Scores (`score_0_1`)

| Metric | `hw500_0014` (human) | `hw500_0050` (human) | `hw500_0150` (human) | `hw500_0363` (HandsOn) | `hw500_0431` (TI5) | `hw500_0452` (TI5) |
| --- | --- | --- | --- | --- | --- | --- |
| MUSIQ | 0.5585 | 0.5627 | 0.6638 | 0.7627 | 0.6638 | 0.7276 |
| Visual Plausibility | 0.6108 | 0.6034 | 0.5914 | 0.5658 | 0.6244 | 0.6486 |
| JEPA Similarity | observed | observed | observed | observed | observed | observed |
| Pose Control | 0.7099 (m) | 0.3996 (m) | 0.5929 (m) | 0.4856 (m) | 0.3161 (m) | 0.4077 (m) |
| Keypoint Coverage | 0.7589 (m) | 0.4435 (m) | 0.6592 (m) | 0.6478 (m) | 0.4046 (m) | 0.5787 (m) |
| Hand Morphology | 0.9503 | 0.8725 | 0.8543 | n/a | n/a | n/a |
| Entity Consistency | 1.0000 | 1.0000 (p) | 1.0000 | 1.0000 (p) | 1.0000 (p) | 1.0000 (p) |
| Background Consistency | 0.9471 | 0.8962 | 0.9162 | 0.8690 | 0.8780 | 0.8761 |
| Photometric Consistency | 0.9281 | 0.7559 | 0.6119 | 0.7216 | 0.8668 | 0.8004 |
| Action Adherence | 0.4000 | 0.8000 | 0.8000 | 0.6000 | 1.0000 | 1.0000 |
| Event Adherence | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Trajectory Accuracy | 1.0000 | 1.0000 (p) | 1.0000 | 1.0000 (p) | 1.0000 (p) | 1.0000 (p) |
| Causal Plausibility | 1.0000 | 0.6667 | 0.8333 | 0.3333 | 0.3333 | 0.6667 |
| Evolution Correctness | 0.5000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 |
| State Regulation | 1.0000 | n/a | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| State Progression | 0.5000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 |
| Memory Consistency | n/a | 0.0000 | n/a | 1.0000 | n/a | n/a |

Slot status: `ok` 68, `scored_with_missing_penalty` 12, `not_applicable` 8, `partial_reference_population` 8, `observed` 6. `(m)` = `scored_with_missing_penalty` (a hand or keypoint that was not found received the
metric's penalty), `(p)` = `partial_reference_population`; both are scored slots and enter the means in
`summary.json`. `n/a` slots (`not_applicable`) have no score and are never averaged. JEPA Similarity is a
collection-level metric (status `observed` per case); its value is in `summary.json` (1.0 for the reference videos).
Reference videos do not score 1.0 on every metric because the judges and detectors are imperfect; for example,
Memory Consistency of `hw500_0050` is 0 because the judge cannot identify the tracked feature in the first
observation window.

HWBScore over the six cases: **74.407** (human 74.235, HandsOn 63.995, TI5 72.629;
metrics not applicable to a whole scope are excluded, see [§9](../../README.md#9-outputs)).

## How close your numbers should be

* `status` must match slot by slot. A slot with `execution_error` means a worker failed; see `logs/` in the run
  directory and run `resume`.
* **VLM-judge metrics** (Visual Plausibility, Action / Event Adherence, Causal Plausibility, Evolution Correctness,
  State Regulation / Progression, Memory Consistency) sample from vision-language judges and may vary slightly across
  hardware and library versions: a judge answer near a decision boundary can flip, and with coarse answer scales one
  changed answer moves a score by a whole step (e.g. 0.8 → 0.6). The same GPU type, driver and library versions
  reproduce the values. Compare models on the full benchmark, not on these six cases.
* **Hand-based metrics** (Pose Control, Keypoint Coverage, Hand Morphology) depend slightly on the scheduling of the
  run: hand detections are batched and shared with the other metrics that run at the same time, so the metric
  selection, the number of GPUs and the task order change the last digits (typically 1e-5 to 5e-4, up to about 2e-3).
* All other metrics agree to about 1e-5 on the same hardware.
