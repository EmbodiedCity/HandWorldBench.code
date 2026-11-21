# Submitting rectified, cropped or resized videos

Use this when your model generated a video from a HandWorldBench first frame and prompt, but the output is not in
the reference camera's native pixels and timeline — for example you undistorted the fisheye input, cropped, resized,
padded, or changed the frame rate. You describe what you did with a **spatial mapping** (reference pixels → your
video's pixels) and a **time mapping** (reference seconds → your video's seconds); the evaluator then moves the
reference hand points and annotation windows into your video's frame before scoring.

If your video has the reference resolution and frame count, you do not need any of this: put `<case_id>.mp4` in a
folder and pass `--videos` (see [the evaluator guide](../../README.md#7-evaluate-your-model)).

## Workflow

1. Take the case's `first_frames/<case_id>.png` and the prompt (`prompts.jsonl` → `prompt`). If you transform the input image (and any image-space control) into your model's viewport,
   record the source calibration, rectifying rotation, final intrinsics, image size and frame correspondence.
2. Generate the video. Include every later crop, resize, pad and frame-rate or timing change in the mapping.
3. Write one prediction record per case with `build_submission.py`, then merge the records of all cases of one model
   into a single JSON array (each case once).

## Spatial mapping

Direction: reference-video pixels → final-video pixels. One JSON object:

| What you did | `kind` | Fields |
| --- | --- | --- |
| nothing (same view, same size) | `identity` | none |
| crop, full-frame resize, or pad | `affine` | `matrix`: 2×3 or 3×3 (a 2×3 matrix gets the row `[0, 0, 1]`; both are equivalent) |
| pinhole undistortion | `opencv_calibrated` | `spec` with `distortion_model: "opencv_pinhole"` |
| equidistant fisheye → pinhole | `fisheye` | `spec` with `distortion_model: "opencv_fisheye_equidistant"` |

A calibrated `spec` has `distortion_model`, `source_wh`, `output_wh`, `K_source`, `D_source`, `R_source_to_target`
and `K_target`. Sizes are `[width, height]`. Pinhole distortion takes 4, 5, 8, 12 or 14 coefficients, fisheye takes 4.
Intrinsics and rotation are 3×3. The source calibration must be the reference video's own (`camera/<case_id>.npz`
→ `K`, `D`). If `K_rect` is the rectified intrinsics and `A` the affine from the rectified image to the final video,
then `K_target = A @ K_rect`.

Example: crop box `[x, y, w, h]` resized to `[W, H]`, with `sx = W/w`, `sy = H/h` and pixel-centre convention:
`A = [[sx, 0, (0.5-x)*sx-0.5], [0, sy, (0.5-y)*sy-0.5], [0, 0, 1]]`. Add the pad offset if you also padded.

`opencv_calibrated` and `fisheye` need the case's reference camera-space 3-D joints (all 500 cases have them). The
short `opencv` form and the full protocol are in [input formats](../../docs/inputs.md).

## Time mapping

Maps reference seconds to your video's seconds using decoded timestamps, both starting at 0.

* unchanged timeline: `{"kind": "identity"}`
* constant speed change: `{"kind": "affine", "scale": 1.0, "offset": 0.0}` with your actual scale and offset
* anything else: `{"kind": "points", "points": [[ref_s, out_s], ...]}`, both columns strictly increasing and covering
  the evaluated interval. If you merged windows or dropped frames, use the frames and timestamps you actually wrote.

## Build the record and check it

Run from the repository root (`HandWorldBench/`), with the benchmark downloaded to `HandWorldBench-data/`. Paths in
the record are stored relative to the dataset directory.

```sh
CASE_ID=hw500_0001
python3 evaluation/examples/rectified_submission/build_submission.py \
  --dataset HandWorldBench-data --case-id "$CASE_ID" --model MyModel \
  --video "generated/MyModel/$CASE_ID.mp4" \
  --spatial-mapping generated/MyModel/spatial_mapping.json \
  --time-mapping generated/MyModel/time_mapping.json \
  --output generated/MyModel/predictions.json

sh evaluation/scripts/run_evaluation.sh prepare \
  --dataset HandWorldBench-data --model MyModel \
  --predictions generated/MyModel/predictions.json --run-dir runs/MyModel
```

`prepare` validates the inputs and builds the task graph without running any model; `reports/prepare.json` lists
binding errors per case. When it is clean, run the same command with `evaluate` instead of `prepare`.
`build_submission.py` refuses to overwrite an existing `--output` file.
