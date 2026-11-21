# Control-signal rendering

This directory renders the three video control signals of HandWorldBench from the hand-joint and camera arrays of a
clip. It is the code that produced the `controls/` videos shipped with the
[HandWorldBench benchmark](https://huggingface.co/datasets/EmbodiedCity/HandWorldBench), and it runs on the
[HandWorldBench-Corpus](https://huggingface.co/datasets/EmbodiedCity/HandWorldBench-Corpus) clips too.

| Signal | What is drawn | Human (egocentric) | HandsOn (simulated Wuji hand) | TI5 (real robot) |
| --- | --- | --- | --- | --- |
| `skeleton` | 21 projected joints and 20 bones per hand | joints | joints | joints |
| `surface` | front-most surface of a hand mesh, flat shading | MANO fitted to the joints | Wuji hand surface (forward kinematics) | spheres and tubes on the joints |
| `xray` | all surface layers of the same mesh, alpha by thickness | same mesh | same mesh | same mesh |

Every video uses the reference camera: same intrinsics and lens model (equidistant fisheye or OpenCV rational),
resolution, frame rate and frame count as the reference video, so control frame *i* aligns with reference frame *i*.
Left hand blue, right hand orange, black background. Everything runs on the CPU; no GPU and no neural network are
needed.

## Contents

| File | Purpose |
| --- | --- |
| `render_controls.py` | command-line entry point: selects cases, prepares hand meshes, runs worker processes |
| `render_case.py` | renders one case: projection, skeleton drawing, mesh rasteriser, surface / X-ray compositing, H.264 encoding |
| `mano_fit.py` | fits MANO to the 3-D joints of a human clip (every other frame) |
| `handson_ik.py` | recovers the Wuji joint angles and palm pose from the 21 published HandsOn joints |
| `robot.py` | Wuji hand forward kinematics and link surfaces |
| `datasets.py` | readers for the benchmark and corpus layouts |
| `assets/{left,right}_landmarks.xml` | Wuji hand MJCF kinematics with the 21 HandWorldBench landmark sites |

## Installation

Requirements: Linux, Python 3.12, `ffmpeg` and `ffprobe` with libx264 on `PATH` (e.g. `apt install ffmpeg`).

```bash
cd data_pipeline/controls
python3.12 -m venv .venv && . .venv/bin/activate
pip install numpy==1.26.4 setuptools wheel
pip install --no-build-isolation -r requirements.txt
```

`chumpy` (needed only to read the official MANO files) does not build in an isolated build environment, hence the
two-step install. `torch` is used on the CPU only; the CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`)
works as well.

### Third-party assets

These are not redistributed; fetch them once.

| Component | Source | Version | Licence | Needed for |
| --- | --- | --- | --- | --- |
| MANO hand model | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (register, download *Models & Code*) | v1.2, `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` | MANO licence, non-commercial research | human `surface` / `xray` |
| Wuji hand description | [github.com/wuji-technology/wuji-hand-description](https://github.com/wuji-technology/wuji-hand-description) | commit `bbc2f64eec834aae616eacc2db1f69016f860c2d` (v2026.04.27) | MIT | HandsOn `surface` / `xray` |
| smplx | [PyPI `smplx`](https://pypi.org/project/smplx/) ([github.com/vchoutas/smplx](https://github.com/vchoutas/smplx)) | 0.1.28 | SMPL-X model licence (non-commercial research) | MANO fitting |
| chumpy | [PyPI `chumpy`](https://pypi.org/project/chumpy/) | 0.70 | MIT | reading the MANO `.pkl` files |

```bash
mkdir -p third_party/mano
# MANO: unzip the download and copy the two model files
cp /path/to/mano_v1_2/models/MANO_LEFT.pkl /path/to/mano_v1_2/models/MANO_RIGHT.pkl third_party/mano/

# Wuji hand meshes
git clone https://github.com/wuji-technology/wuji-hand-description third_party/wuji-hand-description
git -C third_party/wuji-hand-description checkout bbc2f64eec834aae616eacc2db1f69016f860c2d
```

Skeleton and TI5 rendering need neither asset, but the command line asks for each one only when a selected case
needs it (MANO for human cases, Wuji meshes for HandsOn cases).

## Quick start (benchmark)

```bash
hf download EmbodiedCity/HandWorldBench --repo-type dataset --local-dir HandWorldBench-data

python render_controls.py \
    --data-root HandWorldBench-data \
    --out outputs/controls \
    --mano-dir third_party/mano \
    --wuji-meshes third_party/wuji-hand-description/meshes \
    --cases hw500_0002 hw500_0324 hw500_0441
```

If you already downloaded the benchmark for the evaluator, skip the download and pass
`--data-root ../../HandWorldBench-data` instead.

Output:

```
outputs/controls/
├── skeleton/hw500_0002.mp4  surface/hw500_0002.mp4  xray/hw500_0002.mp4   (same for every case)
├── receipts/hw500_0002.json   frame count, timing, projection check, SHA-256 of the three videos
└── work/mano/hw500_0002.npz   cached MANO fit (human cases)
```

Compare with the shipped videos in `HandWorldBench-data/controls/{skeleton,surface,xray}/`. With the pinned
packages and the same libx264 build (ffmpeg 6.1.1, x264 core 164) the files are identical byte for byte; other
encoder builds give the same pixels up to H.264 coding differences.

## Usage

```
python render_controls.py --data-root DIR --out DIR (--cases ID ... | --cases-file FILE | --all)
                          [--layout benchmark|corpus] [--embodiment human|handson|ti5]
                          [--mano-dir DIR] [--wuji-meshes DIR] [--work-dir DIR] [--tmp-dir DIR]
                          [--workers N] [--fit-threads N] [--shard I --num-shards N]
                          [--overwrite] [--previews]
```

| Option | Meaning |
| --- | --- |
| `--data-root` | benchmark root (contains `catalog.json`) or corpus root (contains `manifest.jsonl.gz`); auto-detected |
| `--cases`, `--cases-file`, `--all` | which cases to render; `--embodiment` filters the selection |
| `--mano-dir` | folder with `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` |
| `--wuji-meshes` | the `meshes/` folder of `wuji-hand-description` |
| `--work-dir` | cache for MANO fits (default `<out>/work`); a fit is reused on later runs |
| `--tmp-dir` | where the MP4 files are encoded before being copied to `--out` (default `$TMPDIR`); use a local disk, since some network filesystems reject the writes MP4 muxing needs |
| `--workers` | worker processes (default 4); each case runs in one process |
| `--fit-threads` | PyTorch CPU threads per MANO fit (default 6) |
| `--shard`, `--num-shards` | split the selected cases across machines: machine *i* of *N* renders every *N*-th case starting at *i* |
| `--overwrite` | re-render cases that already have all three videos and a receipt (default: skip them) |
| `--previews` | also write up to five overlay JPEGs per case (reference, skeleton, surface, X-ray side by side) to `<out>/previews/<id>/` |

The command prints one JSON line per case (`ok`, `exists` or `failed` with the traceback) and exits non-zero if any
case failed.

### All 500 benchmark cases

```bash
python render_controls.py --data-root HandWorldBench-data --out outputs/controls \
    --mano-dir third_party/mano --wuji-meshes third_party/wuji-hand-description/meshes \
    --all --workers 16 --tmp-dir /tmp
```

Rendering is CPU-bound. Rough cost per case on one core: a 1,121-frame 1920×1080 human clip about 1–1.5 min
(including a ~35 s MANO fit on 6 threads), HandsOn and TI5 clips 10 s to 3 min depending on length. Plan
`--workers` × `--fit-threads` ≤ the number of cores. To use several machines, run the same command with
`--shard i --num-shards N` on each; outputs from all shards can be written to one shared folder.

### Corpus clips

Download the manifest and the shards you need, and extract each shard into the dataset root so that the paths in
the manifest resolve:

```bash
hf download EmbodiedCity/HandWorldBench-Corpus --repo-type dataset --local-dir HandWorldBench-Corpus \
    --include "manifest.jsonl.gz" "shards/ego-0000.tar"
tar -xf HandWorldBench-Corpus/shards/ego-0000.tar -C HandWorldBench-Corpus

python render_controls.py --data-root HandWorldBench-Corpus --out outputs/corpus_controls \
    --mano-dir third_party/mano --wuji-meshes third_party/wuji-hand-description/meshes \
    --cases hwc_000001 hwc_000002
```

Corpus domains map to embodiments as `ego` → human, `handson` → HandsOn, `ti5` → TI5. A clip whose shard has not
been extracted fails with a message naming the shard. The corpus does not ship rendered controls; these commands
produce them with the same code as the benchmark.

## Input format

Per case the renderer reads (hand 0 = left, hand 1 = right):

| File | Arrays used |
| --- | --- |
| `hands/<id>.npz` | `joints_camera_m` (N, 2, 21, 3) camera-space joints in metres; `joints_2d_px` (N, 2, 21, 2) their projections in pixels; `valid` (N, 2, 21) |
| `camera/<id>.npz` | `K` (3, 3); `D` distortion coefficients; `distortion_model` (`equidistant_fisheye` / `equidistant` → `cv2.fisheye`, otherwise `cv2.projectPoints`) |
| benchmark `catalog.json` | `reference.{width,height,fps,frame_count,video}`, `category`, `geometry.path` |
| corpus `manifest.jsonl.gz` | `domain`, `frames`, `fps`, `video.path`, `hands.path`, `camera.path` (width and height are read from the video) |

The joint order is wrist, then thumb, index, middle, ring and pinky with four joints each (base to tip), as listed
in `geometry.joint_names` of the benchmark catalog. A hand is drawn on frame *t* if `valid[t, hand]` has at least
one true joint; individual joints are drawn when they are finite, in front of the camera (z > 2 cm) and inside the
image.

Before rendering, every case is checked: the reference video (if present) must match the catalog size, frame rate
and frame count, and `joints_2d_px` must be the projection of `joints_camera_m` with `K`, `D` (99th percentile
error ≤ 1.1 px).

## How the meshes are built

- **Human.** For each hand, MANO (45-D pose, flat mean hand, one shape vector per clip) is fitted to the
  root-relative joints of frames 0, 2, 4, …: Kabsch initialisation of the global rotation from the wrist and finger
  bases, then 650 Adam steps (lr 0.03) on the mean squared joint error; the fitted mesh is translated to the observed
  wrist. A frame without its own fit takes the two neighbouring fits (at most 3 frames away), deforms each to the
  true joints of that frame by rotating every bone onto its target direction (vertices follow their three nearest
  bones), and blends them linearly in time.
- **HandsOn.** The 21 joints are the landmarks of the Wuji hand model. `handson_ik.py` recovers the palm pose
  (Kabsch on the six palm landmarks) and the 20 joint angles (Gauss–Newton) exactly, and `robot.py` poses the
  official Wuji link meshes.
- **TI5.** A sphere on every joint (radius 10 mm wrist, 6.5 mm others) and a tube along every bone (8 / 5.5 mm).

Mesh vertices are projected with the clip's lens model and rasterised with a z-buffer (`render_case.raster`).
`surface` keeps the nearest layer with Lambert shading; `xray` composites both hands front to back with
alpha = 0.24 + 0.42·(1 − e^(−thickness/23 mm)) + 0.18·shading. Videos are H.264 (`libx264 -preset veryfast -crf 18`,
yuv420p, no audio, no metadata).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError: ffprobe` / `ffmpeg` | install ffmpeg with libx264 and put it on `PATH` |
| chumpy fails to build (`No module named 'numpy'` / `'pip'`) | install numpy first and use `pip install --no-build-isolation -r requirements.txt` |
| `human cases need --mano-dir` / `FileNotFoundError: .../MANO_RIGHT.pkl` | pass the folder that directly contains `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` |
| `Wuji hand meshes not found` | `--wuji-meshes` must point at `wuji-hand-description/meshes` (with `left/` and `right/`) |
| `projection p99 ...` | `hands/` and `camera/` arrays of the case do not belong together (e.g. a modified or mixed download) |
| `reference video does not match the case metadata` | incomplete or re-encoded reference video; re-download it, or remove it to skip the check |
| `visible hand without reconstructible mesh` | a human hand is visible on a frame with no MANO fit within 3 frames; the fit cache in `--work-dir` may come from different data — delete it |
| MP4 write errors on network storage | pass `--tmp-dir` on a local disk |
| first case slow | numba compiles the rasteriser once and caches it next to `render_case.py`; set `NUMBA_CACHE_DIR` if that folder is read-only |

## Licence

The code in this directory is released under the repository's MIT licence. `assets/{left,right}_landmarks.xml`
are derived from the MuJoCo models of
[wuji-hand-description](https://github.com/wuji-technology/wuji-hand-description) (MIT, © 2025 Wuji Technology),
with the 21 landmark sites added. MANO and smplx are licensed by the Max Planck Institute for Intelligent Systems for
non-commercial research only; rendering human surface and X-ray controls is subject to those terms.
