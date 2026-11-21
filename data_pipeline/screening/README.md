# Clip screening

Scripts that cut continuous clips from the three raw sources and check their technical quality before captioning.
They produced the clip lists of both datasets: 10,000 human (Ego), 6,000 HandsOn and 4,000 TI5 clips for the
corpus, with the 500 benchmark cases held out.

| Source | Size | Main rule |
| --- | ---: | --- |
| Ego (egocentric human, MCAP recordings) | 10,000 | start from a fixed clip list and add the longest clips that pass verification; exclude benchmark recordings, overlapping intervals and duplicates |
| HandsOn (simulation) | 6,000 | match the native length histogram; require three decodable aligned views, front-image checks, render-metadata checksums and finite 52-D state and action |
| TI5 (real robot) | 4,000 | prefer clips that pass the RGB background-motion proxy, then insufficient evidence, then motion-relaxed clips, within native length quotas; keep the HDF5 invalid-value and synchronisation masks |

Every corpus clip is a continuous source interval `[start, end)` of `80n + 1` frames. The robot sources round each episode
down to this form, keep one continuous prefix per episode, compute their length histogram before quality selection
and allocate by largest remainder; a shortage is reported, not filled.

## Installation

Python 3.12 (the tested version) on Linux or macOS, `ffmpeg` / `ffprobe` on `PATH`, and an OpenCV build with FFmpeg (needed for
TI5). From the repository root:

```bash
python -m venv .venv-pipeline && . .venv-pipeline/bin/activate
python -m pip install -r data_pipeline/annotation/requirements.txt
```

The same environment serves screening and annotation. Check it with the bundled tests (CPU, < 1 s):

```bash
cd data_pipeline/screening && python -B test_robot_export.py && cd ../..
```

The test encodes synthetic clips with FFmpeg and checks the robot export's frame mapping and receipts.

## Inputs

The raw sources are not redistributed; you need your own copies. Paths in every manifest are relative to the
manifest file.

| Source | Needs |
| --- | --- |
| Ego | an inventory (one MCAP path per line, relative to the inventory), a recording policy JSON (`scope: whole_recording`, `user_confirmed: true`, `policy_id`, `recordings_file` + `recordings_sha256` for the excluded recording ids), the frozen manifest and its SHA-256, and optionally fingerprint banks of the frozen clips and the benchmark holdout |
| HandsOn | the official `source_manifest.jsonl`, an inventory with decoded frame counts, front and wrist MP4s, native Parquet data, the render-metadata archive, a holdout JSON |
| TI5 | raw folders with `meta.json`, the named videos including the head view, the episode HDF5, a holdout JSON |
| Ego aesthetics (ranking only) | `ViT-L-14.pt` and the LAION aesthetic MLP (`aesthetic_mlp.pth`) in one folder, set as `aesthetics.assets` (and the interpreter as `aesthetics.python`) in the profile JSON |

A holdout JSON has a `records` list: HandsOn records use `category: handson`, `episode_uid`, `raw_video_sha256`;
TI5 records use `category: real_robot`, `source_meta` (relative to the TI5 root) and `raw_video_sha256`.

## Usage

The commands below need the raw source data; every script prints its full option list with `--help`.

### Ego

```bash
python data_pipeline/screening/screen_ego.py \
  --inventory inputs/ego-mcap.txt --recording-policy inputs/recording-policy.json \
  --frozen inputs/ego-frozen.jsonl --frozen-sha256 "$EGO_FROZEN_SHA256" \
  --frozen-pixels inputs/ego-frozen-pixels.jsonl --holdout-pixels inputs/holdout-pixels.jsonl \
  --out runs/ego-verified

python data_pipeline/screening/freeze_ego_release.py \
  --frozen inputs/ego-frozen.jsonl --frozen-sha256 "$EGO_FROZEN_SHA256" \
  --verified runs/ego-verified/manifest.verified.jsonl --verified-sha256 "$EGO_VERIFIED_SHA256" \
  --isolation-policy inputs/recording-policy.json --out runs/ego
```

`screen_ego.py` applies the combined geometry and RGB profile (per-frame hand visibility, native camera motion, image
stability), checks export frame order, decoded-RGB identity, calibration-aligned annotations and the conditioning
frame, and writes `manifest.verified.jsonl` (its digest is also in `runs/ego-verified/COMPLETE.json`).
`freeze_ego_release.py` writes `manifest.ego.jsonl` (the 10,000-clip list) and `manifest.batch02.jsonl` (the
added clips only). Additions need at least 161 frames under the default profile.

Quality profiles: the default `ego_qc.json` is profile `relaxed6_combined_v3`; four alternative profiles are in
`ego_profiles/` (`support50_v3`, `speedp95_v3`, `moderate4_residual_v1`, `relaxed6_soft`) and are selected with
`--config`. Each run uses one profile on its input list.

Fingerprint banks hold `id` and three 64-bit `fp` values (first, middle, last decoded frame); frozen receipts also bind
`video_sha` and `source_sha`. With both banks the runner reproduces the release's visual-overlap checks; without them
it still checks recording isolation, frozen-interval overlap, exact video identity and within-run three-frame
similarity, and reports how many external fingerprints it compared.

### HandsOn and TI5

```bash
python data_pipeline/screening/rescreen_robot.py --domain handson \
  --source-root inputs/handson --inventory inputs/handson-inventory.jsonl \
  --holdout inputs/robot-holdout.json --root runs/handson

python data_pipeline/screening/rescreen_robot.py --domain ti5 \
  --source-root inputs/ti5 --holdout inputs/robot-holdout.json --root runs/ti5-technical

python data_pipeline/screening/ti5_proxy_scan.py \
  --inventory runs/ti5-technical/reference.inventory.jsonl --root runs/ti5-proxy

python data_pipeline/screening/ti5_select.py \
  --base runs/ti5-technical --proxy runs/ti5-proxy --out runs/ti5
```

Outputs: `runs/handson/manifest.native_distribution.jsonl` and `runs/ti5/manifest.ti5.jsonl`. For large TI5 runs use
`--shards N --shard I` per index and finalise once the proxy receipts cover the same inventory. The camera proxy is
an RGB background registration in image-plane units; camera pose itself is not measured. Robot exports assign
playback timestamps from decoded source-frame indices and record the mapping and frame rate per view.

### Resume

Re-running a command on the same output directory resumes it. The robot screen and the TI5 selector re-check source
identity, fingerprints, every exported view and the first image, and regenerate clips whose outputs are missing,
changed, or lack the frame-mapping receipt. If a regenerated video's SHA-256 changes, re-check downstream caption and
geometry manifests before reusing them. A changed input or policy stops the run ("use a new run directory").

## Files

| File | Role |
| --- | --- |
| `screen_ego.py`, `freeze_ego_release.py`, `ego_pixels.py`, `ego_qc.json`, `ego_profiles/` | Ego screening, freezing, pixel fingerprints, profiles |
| `handclip/` | Ego clip curation library (temporal windows, amplitude, recording isolation, release assembly) |
| `rescreen_robot.py`, `robot_io.py` | HandsOn / TI5 screening and export |
| `sim_screen.py`, `ti5_screen.py`, `ti5_inventory.py` | per-domain helpers imported by `rescreen_robot.py` |
| `ti5_proxy_scan.py`, `ti5_select.py`, `ti5_camera_policy.json` | TI5 background-motion proxy and tiered selection |
| `test_robot_export.py` | unit test |

Next stage: [annotation](../annotation/README.md).
