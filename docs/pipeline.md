# Pipeline

Three stages, each consuming the previous stage's output. Every stage is
independently runnable and records what it checked, so a failure is localised.

```
raw recordings
    |
    v
screening/  -> manifest.*.jsonl          one row per accepted clip:
    |                                         source interval, frame count,
    |                                         video digest, check receipts
    v
annotation/ -> <merged>.jsonl                 the same rows plus:
    |                                         task_text, segments, operation,
    |                                         caption_path, caption_sha256
    v
controls/   -> controls/{skeleton,surface,xray}/<id>.mp4
                                               rendered from the hands/ and camera/
                                               arrays (data, not produced here)
    |
    v
 (a model generates a video; the evaluator scores it)
```

## Contracts between stages

**screening → annotation.** A row carries `id`, `domain`, `frames`, `fps`,
`source_frames` (half-open), `video` and `video_sha`. Annotation refuses a row
whose video digest does not match the file it opens, so a re-encoded video
invalidates its captions.

**arrays → controls.** The renderer reads the dataset record directly — `catalog.json` (benchmark: size, frame
rate, frame count) or `manifest.jsonl.gz` (corpus: frame rate and frame count; the frame size is read from the clip
video) — together with the clip's `hands/<id>.npz` (joints) and `camera/<id>.npz` (intrinsics, distortion model);
captions are not needed. A clip without `hands` and
`camera` arrays is still valid for captioning but cannot produce controls. See
[data_pipeline/controls/README.md](../data_pipeline/controls/README.md) for how to render the skeleton, surface and
X-ray videos.

**controls → evaluation.** A generated video is submitted with its case id and,
if the model changed the geometry, the spatial and temporal mapping it applied.
The evaluator compares against the same clip id in the evaluation bundle.

## The evaluation bundle

The 500 evaluation cases are held out from the corpus. Their human clips follow the same clip rules as the corpus
(continuous `80n + 1`-frame intervals) and come from recordings that contribute no corpus clips; their HandsOn and TI5
cases are native episode segments. Their captions and operation labels come from the annotation stage but are stored
in a different layout (`operation_taxonomy`, see [data_format.md](data_format.md)), and each case also carries the
metric annotations the evaluator reads. The two bundles share the array schema, which is what lets the same loader
and the same control-signal scripts work on both.

## Stage independence

| Stage | Needs GPU | Needs models | Can be skipped |
| --- | --- | --- | --- |
| screening | no | no | if inputs are already screened |
| annotation | yes (vLLM) | one vision-language checkpoint | if captions exist |
| controls – skeleton | no (CPU) | no | — |
| controls – surface/x-ray | no (CPU) | human: MANO model files; HandsOn: public Wuji hand meshes (MIT); TI5: nothing extra | if only skeleton is wanted |
| evaluation | yes | the metric model zoo | — |

Nothing in the pipeline runs a generation model. The pipeline stops at control
signals; producing a video is outside its scope.
