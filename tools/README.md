# Tools

Two stand-alone utilities (Python 3.12, the tested version, and NumPy; the camera renderer also needs OpenCV and `ffmpeg`).

## `check_handoffs.py` — consistency checks between stages

Checks that two manifests agree before you hand data to the next stage. It reads JSON arrays, objects with a
`records` / `predictions` / `cases` list, or JSONL.

| Check | Arguments | Verifies |
| --- | --- | --- |
| training handoff | `--captions <merged caption manifest> --controls <control manifest>` | every control record has a caption record (`id` / `case_id` / `clip_id`), no duplicates, and, where a control record carries `original_video_sha256`, that it equals the caption source's `video_sha` |
| benchmark handoff | `--predictions <prediction manifest> --catalog <catalog.json>` | every prediction names a benchmark case and a video, and no (model, case) pair repeats |

```bash
python tools/check_handoffs.py --predictions predictions.json --catalog HandWorldBench-data/catalog.json
```

prints, for a manifest with three predictions,

```json
{
  "benchmark": {
    "catalog_cases": 500,
    "prediction_records": 3,
    "prediction_cases": 3
  }
}
```

and exits with status 2 and a one-line reason on the first inconsistency.

## `render_camera_track.py` — draw a clip's camera trajectory

Renders the per-frame camera pose of `camera/<id>.npz` (`c2w`) as a moving frustum, optionally over the first frame.
Visualisation only — models consume `c2w` directly.

```bash
python tools/render_camera_track.py --camera HandWorldBench-data/camera/hw500_0001.npz \
  --first-frame HandWorldBench-data/first_frames/hw500_0001.png --out track_hw500_0001.mp4
```

It prints `{"out": ..., "frames": N}` when done (OpenCV may first log that a hardware H.264 encoder is unavailable; the
file is still written). Options: `--fps` (default 30), `--stride` (draw every n-th pose, default 1). HandsOn cameras are fixed per episode and
TI5 cameras are fixed relative to frame 0, so their tracks are a single pose.
