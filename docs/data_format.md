# Data format

Both datasets store per-frame hand and camera arrays in `.npz` files with the same key names, so code written against
one reads the other. They differ in the per-clip record (benchmark `catalog.json`, corpus `manifest.jsonl.gz`), in a
few vocabulary labels and in the dtype of the point arrays. The tables below were generated from the released files
(one clip per embodiment and dataset).

## Vocabulary

| | Benchmark (`HandWorldBench`) | Corpus (`HandWorldBench-Corpus`) |
| --- | --- | --- |
| embodiment field | `category`: `human`, `handson`, `real_robot` | `domain`: `ego`, `handson`, `ti5` |
| ids | `hw500_0001`–`0500` | `hwc_000001`–`020016` (20,000 ids) |
| human camera model | `equidistant_fisheye` (OpenCV fisheye, 4 coefficients) | `equidistant` (same model, same 4 coefficients) |
| HandsOn camera model | `opencv_rational` (OpenCV, 8 coefficients) | `opencv` (same model, same 8 coefficients) |
| TI5 camera model | `opencv_rational` with all-zero coefficients (nominal intrinsics) | `pinhole`, `D` empty (nominal intrinsics) |
| point dtype | float64 (the exact arrays the evaluator scores against; `joints_camera_m` of `hw500_0441` is float32) | float32 |

`equidistant` / `equidistant_fisheye` → `cv2.fisheye.projectPoints`; `opencv` / `opencv_rational` / `pinhole` →
`cv2.projectPoints`. Using the wrong model gives errors of tens to hundreds of pixels.

## Benchmark record: `catalog.json`

One record per case, in the format the evaluator reads (`evaluation/docs/inputs.md`): `case_id`, `category`,
`task_description`, `reference` (`video`, `sha256`, `width`, `height`, `fps`, `frame_count`), `geometry` (which keys of
`hands/<id>.npz` to read: `path`, `points_key`, `camera_points_key`, `time_key`, `positive_depth_key`, `hand_order`,
`joint_names` (all cases except `hw500_0441`), and for robot cases `time_origin`, `joint_indices`,
`landmark_convention`, plus `frame_index_key`, `frame_index_reference_sha256` and `status` for TI5),
`applicability` (17 metrics), `annotations` (`semantic`, `views`, `supplement`, `background`; documents referenced by
path live in `assets/`) and `files` (`first_frame`, `camera`, `annotations`, `controls`). `prompts.jsonl` has one line
per case: `case_id`, `category`, `prompt` (= `task_description`), `first_frame`, `video`, `reference_sha256`,
`width`, `height`, `fps`, `frame_count`. The benchmark dataset card describes every field.

## Corpus record: `manifest.jsonl.gz`

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | `hwc_NNNNNN` |
| `domain` | string | `ego`, `handson` or `ti5` |
| `frames` | int | number of video frames |
| `fps` | float | 30.0 |
| `source_view` | string | `main` (ego), `front` (HandsOn), `head` (TI5) |
| `shard` | string | the tar file that holds the clip's files |
| `video` | object | `{path, sha256}` |
| `first_frame` | object or null | `{path, sha256}`; `null` for 3,636 ego clips |
| `camera` | object | `{path, sha256, view, reference, intrinsics}`; `reference` ∈ `clip_first_frame`, `simulation_world`; `intrinsics` ∈ `source`, `simulation`, `estimated` |
| `hands` | object | `{path, sha256, view, coordinate_frame, joints_3d}`; `coordinate_frame` = `camera`; `joints_3d` ∈ `source`, `simulation`, `estimated` |
| `caption` | object | see below |

Paths are relative to the dataset root; shards are extracted into that root. `index.parquet` repeats `id`, `domain`,
`frames`, `fps`, `source_view`, the caption text, the primary operation code, `shard` and `hand_supervision`.

```json
{"text": "...",
 "segments": [{"frame_span": [0, 27], "text": "...", "evidence_frames": [0, 6, 13, 20, 26]}],
 "evidence_frames": [0, 20, 40, 60, 80],
 "operation": {"code": "M1", "family": "Move & Arrange", "family_id": "move_arrange",
               "subtype": "Pick / Place", "additional": ["O2"]}}
```

`frame_span` is half-open and the spans tile `[0, frames)`. `code` is a sub-type code from `taxonomy.json` or `OTHER`
(`family: null`). The benchmark's `annotations/<id>.json` holds `caption` (`text`, `segments`) and `operation_taxonomy`
(`labels[]` with `family`, `family_id`, `subtype`, `subtype_id`, `role` ∈ primary / additional, plus `primary_*`
fields; ids as in `operation_taxonomy.json`, not the corpus codes).

## `hands/<id>.npz`

| Key | Benchmark | Corpus | Meaning |
| --- | --- | --- | --- |
| `joints_2d_px` | (T, 2, 21, 2) float64 | (T, 2, 21, 2) float32 | joints in video pixels |
| `joints_camera_m` | (T, 2, 21, 3) float64 | (T, 2, 21, 3) float32 | joints in the camera frame, metres |
| `valid` | (T, 2, 21) bool | (T, 2, 21) bool | 3-D point available |
| `valid_uv` | (T, 2, 21) bool | (T, 2, 21) bool | 2-D point usable (finite, in front of the camera, hand valid) |
| `in_frame_uv` | (T, 2, 21) bool | (T, 2, 21) bool | `valid_uv` and inside the image |
| `t_s` | (T,) float64 | (T,) float64 | seconds; 0 at the first sample of every clip |
| `frame_index` | (T,) int64 | (T,) int32 | benchmark human: frame index in the source recording; otherwise the video frame of each sample |
| `camera_K`, `camera_distortion` | benchmark HandsOn and TI5 only: (3, 3), (8,) (5 for `hw500_0441`) float64 | — | the intrinsics the evaluator checks for calibrated spatial mappings |

Hands are ordered **left, right**. Each hand has 21 points:

```
0      wrist / palm anchor (benchmark catalog.json → geometry.joint_names: "wrist" for
       human cases, "palm_anchor" for robot cases)
1..4   thumb: base, middle, distal, tip
5..8   index
9..12  middle
13..16 ring
17..20 little finger
```

## `camera/<id>.npz`

| Key | Shape, dtype | Meaning |
| --- | --- | --- |
| `K` | (3, 3) float64 | intrinsics at the video resolution |
| `D` | (N,) float64 | distortion coefficients: 4 (fisheye), 8 (OpenCV rational), 5 zeros (`hw500_0441`), 0 (corpus TI5) |
| `distortion_model` | () str | see [Vocabulary](#vocabulary) |
| `c2w` | (T, 4, 4) float64 | OpenCV camera-to-world pose |
| `valid` | (T,) bool | pose usable |
| `t_s` | (T,) float64 | seconds, same origin as the hands file |
| `frame_index` | (T,) int32 | benchmark human: frame index in the source recording (as in the hands file); otherwise the video frame of each sample |

`c2w` maps `X_cam` to `X_world = R @ X_cam + t`. The world frame is: benchmark human — the capture rig's world frame of
the recording (`c2w[0]` is generally not the identity); corpus ego — the first frame of the clip
(`camera.reference = clip_first_frame`); HandsOn — the simulation world, constant within a clip; TI5 — the camera is
fixed and world extrinsics are unavailable, so `c2w` is the identity.

TI5 intrinsics are nominal estimates (principal point at the image centre, no distortion). Corpus TI5 clips all use
fx = 521.513027, fy = 530.453226, cx = 320, cy = 240; benchmark TI5 cases carry their own nominal values. TI5 3-D
joints rely on estimated depth.

Projection of camera-space points:

```python
import cv2, numpy as np
zero = np.zeros(3)
if model in ("equidistant", "equidistant_fisheye"):
    uv, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), zero, zero, K, D)
else:  # opencv, opencv_rational, pinhole
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), zero, zero, K, D if D.size else None)
```

## Identifiers

The two datasets use different id spaces and share no video. Source identifiers are not published.
