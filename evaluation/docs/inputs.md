# Input formats

## Dataset

`--dataset` points to a directory containing `dataset.json` and a catalog:

```json
{
  "catalog": "catalog.json",
  "standard_output_protocol": {
    "name": "reference_native",
    "time_mapping": {"kind": "identity"},
    "spatial_mapping": {"kind": "identity"},
    "match_reference_dimensions": true,
    "match_reference_timeline": true
  }
}
```

The standard protocol describes the actual expected output coordinates and timeline. Individual cases can specify their own `standard_output_protocol`.

Explicit `spatial_mapping` and `time_mapping` fields always take priority. Existing custom dataset protocols remain authoritative. For ordinary inputs, missing mapping fields are completed under the full-frame resize / complete-clip uniform-speed contract below. A matching `reference_native` input retains its existing identity mappings and validation.

A crop, padding, undistortion, or changed camera view requires an explicit mapping through a sidecar or prediction manifest. Image dimensions alone are never used to infer these operations.

`catalog.json` is an array. Each case supplies:

```json
{
  "case_id": "example",
  "category": "handson",
  "reference": {
    "video": "assets/videos/example.mp4",
    "sha256": "<video SHA-256>",
    "width": 640,
    "height": 480,
    "fps": 30,
    "frame_count": 300
  },
  "geometry": {
    "path": "assets/geometry/example.npz",
    "points_key": "joints_uv",
    "camera_points_key": "joints_camera_m",
    "time_key": "source_time_s",
    "time_origin": "first_sample",
    "hand_order": ["left", "right"],
    "joint_indices": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "status": "ok"
  },
  "applicability": {
    "pose_control": "applicable",
    "keypoint_coverage": "applicable",
    "entity_consistency": "applicable"
  },
  "annotations": {
    "semantic": "assets/annotations/example.json",
    "views": [],
    "supplement": {},
    "background": {
      "object_concepts": ["wooden cube"],
      "concepts_complete": true
    }
  }
}
```

The keys above are examples; the released benchmark uses `joints_2d_px` and `t_s` (see its `catalog.json` → `geometry`).

Categories are `human`, `handson`, and `real_robot`. Metric applicability uses canonical metric IDs. `applicable` creates scoring tasks; `not_applicable`, `pending`, and reference/input error statuses remain explicit result slots.

Geometry points have shape `[frames, 2, 21, 2]`; camera points have shape `[frames, 2, 21, 3]`. Point order is wrist/palm anchor followed by the thumb, index, middle, ring, and little-finger chains. `joint_indices` maps the source order to this order. `hand_order` specifies the source hand axis. Optional `positive_depth_key` and `hand_valid_key` identify geometric validity arrays. Reference visibility is not required.

Hand geometry supports pose, coverage, hand-region identity, and morphology evaluation. A geometry `status` of `ok`, `ready`, or `verified` permits these consumers; pending calibration produces `missing_reference` slots for them. Omitting the status uses the supplied asset directly. Trajectory evaluation consumes reference frames and object tracks independently of hand geometry.

## Prediction manifest

`--predictions` accepts an array or an object with a `predictions` array:

```json
[
  {
    "case_id": "example",
    "model": "MyModel",
    "video": "../MyModel/videos/example.mp4",
    "time_mapping": {"kind": "identity"},
    "spatial_mapping": {"kind": "identity"}
  }
]
```

`video_sha256` can supply an existing trusted video fingerprint. Otherwise, preparation computes it once. A per-row `metrics` list can restrict the requested subset further. Every row in a run belongs to the model selected with `--model`; a row may repeat it in `model`. Relative prediction video paths resolve against the dataset directory.

For directory input, a sidecar named `<case_id>.json` accepts the same mapping fields. Multiple video files for one case require an explicit manifest selection. Directory and prediction-manifest inputs use the same missing-field completion rules.

## Time and pixel mappings

Time mappings convert reference seconds into generated-video seconds. Identity, affine, and piecewise mappings are supported by `handworldbench.data.timeline.Timeline`. The same mapping binds keypoint samples and annotation windows.

For an ordinary submission with no time mapping, the evaluator assumes the complete reference interval is present and only one uniform playback-speed change occurred. It reads actual decoded frame timestamps from both videos and maps the first decoded reference frame to the first decoded prediction frame and the last reference frame to the last prediction frame:

`prediction_time = reference_time * prediction_timestamp_span / reference_timestamp_span`

Each video's first timestamp is normalized to zero. Equal timestamp spans use identity even when FPS or frame count differs. Container duration and nominal FPS are not used to infer missing action content, trims, pauses, or nonlinear timing. Both videos require at least two strictly increasing, finite frame timestamps. The evaluator does not perform action matching or DTW.

For a missing spatial mapping, the evaluator assumes a full-frame resize from catalog reference dimensions to decoded prediction dimensions. Equal dimensions use identity. Otherwise `sx = output_width / reference_width`, `sy = output_height / reference_height`, and pixel centers follow `x_out = (x_ref + 0.5) * sx - 0.5` (and similarly for y). This does not imply crop or padding. Automatically bound decoded dimensions and timeline are rechecked during normal decoding. If only one mapping is missing, only that mapping is completed; the explicit other mapping is preserved.

Spatial mappings support identity, homogeneous affine matrices, recorded OpenCV undistortion/resizing, and recorded fisheye-to-pinhole calibration. Fisheye geometry uses reference camera-space 3D points and the recorded target camera. Coordinate transformations follow each video's declared image version.

Binding errors are attached to the affected input; other videos continue. Missing predictions inside a valid evaluation input receive the metric's missing-observation penalty.

## Event, memory, and state annotations

`annotations.semantic` contains action, event, and causal inputs. It may be an inline object or a JSON file path. `annotations.views` binds state specifications, state protocols, memory annotations/checklists, segmentation concepts, and per-view applicability.

`annotations.supplement.event_adherence.annotation` supplies the event-specific semantic annotation. `annotations.supplement.memory_consistency.inputs` supplies explicit memory annotations and `memory_unit_checklists`, including planned return windows. Each supplement is scoped to its own metric.

Preparation fingerprints each metric's bound annotation content. Changing a memory checklist invalidates memory evidence and scoring while retaining unrelated observations. Pending event units remain outside the applicable scoring set.

Source windows are converted to prediction time before evidence extraction. Short windows use their actual available frames. Each video averages its applicable event units equally; video means receive equal weight in model summaries.

## Reference validity and object inventories

Geometry can declare `hand_valid_key` for a boolean `[frames, 2]` array. Invalid
reference hands are excluded from the reference evaluation population; they are
not prediction failures. Use the video's presentation clock for `time_key`.
For fixed-rate source videos this can be `source_frame_index / fps`; acquisition
timestamps need not equal the video's playback clock.

`annotations.background.reference_entities` may declare a stable object inventory
bound to `reference_sha256`, with object `entities` and `localization` containing
`source_frame_indices` and `evidence`. Object locations are still inferred from
each role's own images. Changing an object's location does not create a new ID.

Time mapping points are `[reference_seconds, prediction_seconds]`. Explicit mappings remain appropriate when the full-clip uniform-speed assumption is not valid. For known subsampling or retiming, provide the actual correspondence instead of relabeling the video as native. Omitting mappings selects the full-clip contract; it does not prove that all required actions were generated.


### Frame-indexed geometry

For geometry synchronized to decoded reference frames, declare `frame_index_key` and `frame_index_reference_sha256`. The integer index array must be strictly increasing and bound to the reference video SHA-256. Evaluation selects the geometry row for each decoded reference frame. Without this declaration, the recorded geometry time axis and its tolerance apply.
