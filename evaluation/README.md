# HandWorldBench evaluator

`handworldbench` scores generated hand-interaction videos against the 500 reference cases of the
[HandWorldBench benchmark](https://huggingface.co/datasets/EmbodiedCity/HandWorldBench) with **17 metrics in six
dimensions** (visual quality, hand fidelity, appearance consistency, instruction following, physical interaction,
temporal coherence). Human hands, simulated dexterous hands (HandsOn) and real dexterous hands (TI5) share one
interface. Every score is in **[0, 1], higher is better**; `hwbscore.json` adds a 0–100 summary.

- [1. Requirements](#1-requirements)
- [2. Install the environments](#2-install-the-environments)
- [3. Download the model weights](#3-download-the-model-weights)
- [4. Download the benchmark](#4-download-the-benchmark)
- [5. Bind the runtime configuration](#5-bind-the-runtime-configuration)
- [6. Check the installation](#6-check-the-installation)
- [7. Evaluate your model](#7-evaluate-your-model)
- [8. Monitor, resume and compare runs](#8-monitor-resume-and-compare-runs)
- [9. Outputs](#9-outputs)
- [10. Metrics](#10-metrics)
- [11. Runtime and GPU memory](#11-runtime-and-gpu-memory)
- [12. Troubleshooting](#12-troubleshooting)

Commands marked **[GPU]** need an NVIDIA GPU and the model weights; every other command runs on CPU. The guide was
tested on Linux x86-64 with NVIDIA H20 96 GB GPUs, driver 570, inside the NVIDIA PyTorch 25.03 base environment, with
the weights listed (with checksums) in [`setup/model_resources.json`](setup/model_resources.json).

## 1. Requirements

| | |
| --- | --- |
| OS / GPU | Linux x86-64, NVIDIA GPU with ≥ 80 GB memory for the visual-plausibility model (all other models fit in 40 GB); driver ≥ 570 |
| Base environment | [NVIDIA PyTorch 25.03](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html) (Python 3.12, CUDA 12.8.1, PyTorch 2.7.0a0); the main, SAM3 and metrics environments inherit its PyTorch and torchvision |
| Extra interpreter | Python 3.8 for the MUSIQ worker (JAX 0.4.1 / TensorFlow 2.7 with CUDA 11 libraries) |
| Tools | `ffmpeg`, `ffprobe`, `nvidia-smi`, `git`, `wget`, `unzip` on `PATH` |
| Disk | 105 GB for weights, 8.7 GB for the benchmark, plus caches (≈ 120 MB per evaluated case in `cache/`) |

Directory layout used throughout (all commands run from the repository root):

```text
HandWorldBench/                 # git clone https://github.com/EmbodiedCity/HandWorldBench
├── evaluation/                 # this directory
├── .envs/{main,sam3,metrics,musiq,musiq-cuda11}   # created in step 2
├── models/                     # weights, step 3
└── HandWorldBench-data/        # benchmark, step 4 (default --dataset of the launcher)
```

## 2. Install the environments

Four interpreters keep the pinned `timm`/`transformers` versions of each backend apart. Create them on a local
filesystem: virtual environments need symbolic links, which some network filesystems (e.g. ossfs) do not support.
If the base image sets `PIP_CONSTRAINT`, clear it so the pins below win; keep the image's PyTorch and torchvision.

```bash
git clone https://github.com/EmbodiedCity/HandWorldBench
cd HandWorldBench
unset PIP_CONSTRAINT
python3.12 -m venv --system-site-packages .envs/main
python3.12 -m venv --system-site-packages .envs/sam3
python3.12 -m venv --system-site-packages .envs/metrics
.envs/main/bin/python    -m pip install -r evaluation/setup/requirements.main.txt
.envs/main/bin/python    -m pip install -e evaluation
.envs/sam3/bin/python    -m pip install -r evaluation/setup/requirements.sam3.txt
.envs/metrics/bin/python -m pip install -r evaluation/setup/requirements.metrics.txt
.envs/metrics/bin/python -m pip install --no-deps vjepa==0.1.2
```

`vjepa` is installed without dependency resolution because its metadata asks for another torchvision series; the
metrics requirements already contain what it needs.

MUSIQ runs in its own Python 3.8 process with CUDA 11 / cuDNN 8 libraries. Any Python 3.8 works; one way to get it:

```bash
.envs/main/bin/python -m pip install uv               # the base image's system Python is externally managed (PEP 668)
export UV_PYTHON_INSTALL_DIR=$PWD/.envs/uv-python      # keep the interpreter next to the environments
.envs/main/bin/uv python install 3.8
PY38=$(.envs/main/bin/uv python find 3.8)
"$PY38" -m venv .envs/musiq
.envs/musiq/bin/python -m pip install -r evaluation/setup/requirements.musiq.txt
.envs/main/bin/python -m pip install --no-deps --target .envs/musiq-cuda11 \
  -r evaluation/setup/requirements.musiq-cuda11.txt
```

`requirements.musiq.txt` pulls `jaxlib==0.4.1+cuda11.cudnn86` from the
[JAX CUDA wheel index](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).

Optional: `causal-conv1d==1.7.0` (`.envs/main/bin/python -m pip install --no-build-isolation causal-conv1d==1.7.0`)
speeds up the Qwen3.5 judge. It needs the CUDA compiler and network access to GitHub for its prebuilt wheel; the
evaluator runs without it (the reference outputs in [`examples/expected_outputs/`](examples/expected_outputs/README.md)
were produced without it).

## 3. Download the model weights

Thirteen assets, ≈ 105 GB, all from their public upstream locations. Save each at the path in the second column;
the pinned revisions and SHA-256 values are in [`setup/model_resources.json`](setup/model_resources.json).

| Used by | Path under `models/` | Source |
| --- | --- | --- |
| hand detection (all hand metrics) | `grounding-dino-tiny/` | [IDEA-Research/grounding-dino-tiny @ a2bb814](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71) |
| Pose Control, Keypoint Coverage | `rtmpose-hand/end2end.onnx` | [RTMPose-m Hand5 ONNX zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip) → `end2end.onnx` |
| object localisation | `Qwen3-VL-4B-Instruct/` | [Qwen/Qwen3-VL-4B-Instruct @ ebb281e](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| action, event, causal, state, memory judge | `Qwen3.5-9B/` | [Qwen/Qwen3.5-9B @ c202236](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) |
| hand / object / background masks | `sam3/sam3.pt` | [facebook/sam3 @ 3c879f3](https://huggingface.co/facebook/sam3/blob/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt) — gated: request access on the model page first. The BPE vocabulary ships with this repository. |
| Entity Consistency | `clip/ViT-B-32.pt` | [OpenAI CLIP ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| Background Consistency | `dinov3/dinov3_vits16_pretrain_lvd1689m.pth` | [DINOv3 ViT-S/16 LVD-1689M](https://github.com/facebookresearch/dinov3#pretrained-models) — request the weights from Meta, download `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, save under the name on the left |
| MUSIQ | `musiq/spaq_ckpt.npz` | [Google Research SPAQ checkpoint](https://storage.googleapis.com/gresearch/musiq/spaq_ckpt.npz?generation=1638397900698348) |
| Visual Plausibility | `qwen3vl-a3b-visual-plausibility/` | [meituan-longcat/WBench-weights @ 1c5fde9](https://huggingface.co/meituan-longcat/WBench-weights/tree/1c5fde9aed33c2659a620e244d0151b7741e237d/qwen3vl-a3b-visual-plausibility) |
| Photometric Consistency | `DA3-GIANT-1.1/` | [meituan-longcat/WBench-weights @ 1c5fde9](https://huggingface.co/meituan-longcat/WBench-weights/tree/1c5fde9aed33c2659a620e244d0151b7741e237d/DA3-GIANT-1.1) |
| Hand Morphology | `hand_morphology/vbench2_hand.pth` | [Vchitect/VBench-2.0_models `hand.pth`](https://huggingface.co/Vchitect/VBench-2.0_models/resolve/ebdeff2bd48e113f76db9ad1d5b59c8df3456d8e/Human_Anatomy/anomaly_detector/hand.pth), saved as `vbench2_hand.pth` |
| JEPA Similarity | `jepa/vith16.pth.tar`, `jepa/ssv2-probe.pth.tar` | [V-JEPA ViT-H/16](https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar) · [SSv2 probe](https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar) |

The Hugging Face entries can be fetched with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
`hf` is installed in `.envs/main` (use `.envs/main/bin/hf`, or `pip install -U huggingface_hub`):

```bash
hf download IDEA-Research/grounding-dino-tiny --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 --local-dir models/grounding-dino-tiny
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir models/Qwen3-VL-4B-Instruct
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
hf download facebook/sam3 sam3.pt --revision 3c879f39826c281e95690f02c7821c4de09afae7 --local-dir models/sam3
hf download meituan-longcat/WBench-weights --revision 1c5fde9aed33c2659a620e244d0151b7741e237d \
  --include 'DA3-GIANT-1.1/*' 'qwen3vl-a3b-visual-plausibility/*' --local-dir models
mkdir -p models/hand_morphology models/rtmpose-hand models/clip models/musiq models/jepa models/dinov3 models/classic
wget -O models/hand_morphology/vbench2_hand.pth https://huggingface.co/Vchitect/VBench-2.0_models/resolve/ebdeff2bd48e113f76db9ad1d5b59c8df3456d8e/Human_Anatomy/anomaly_detector/hand.pth
wget -O models/clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget -O models/musiq/spaq_ckpt.npz 'https://storage.googleapis.com/gresearch/musiq/spaq_ckpt.npz?generation=1638397900698348'
wget -P models/jepa https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
wget -O /tmp/rtmpose.zip https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip
unzip -j /tmp/rtmpose.zip '*end2end.onnx' -d models/rtmpose-hand
# DINOv3: after Meta grants access, save the file as models/dinov3/dinov3_vits16_pretrain_lvd1689m.pth
```

Size on disk and access terms (licences: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)):

| Path under `models/` | Size | Access |
| --- | ---: | --- |
| `qwen3vl-a3b-visual-plausibility/` | 58 GB | open |
| `Qwen3.5-9B/` | 19 GB | open |
| `jepa/` (2 files) | 9.9 GB | open, non-commercial (CC BY-NC 4.0) |
| `Qwen3-VL-4B-Instruct/` | 8.3 GB | open |
| `DA3-GIANT-1.1/` | 5.1 GB | open; Depth Anything 3 giant weights are non-commercial upstream |
| `sam3/sam3.pt` | 3.3 GB | **gated** — request access on the Hugging Face page |
| `grounding-dino-tiny/` | 0.66 GB | open |
| `clip/ViT-B-32.pt` | 0.34 GB | open |
| `hand_morphology/vbench2_hand.pth` | 0.33 GB | open |
| `musiq/spaq_ckpt.npz` | 0.16 GB | open |
| `dinov3/…pth` | 0.08 GB | **gated** — request from Meta |
| `rtmpose-hand/end2end.onnx` | 0.05 GB | open |

`models/classic/` may stay empty; the runtime binds every checkpoint explicitly. Step 5 (`--check`) reports any file
that is missing.

## 4. Download the benchmark

```bash
hf download EmbodiedCity/HandWorldBench --repo-type dataset --local-dir HandWorldBench-data
(cd HandWorldBench-data && sha256sum -c SHA256SUMS --quiet)
```

The launcher uses `HandWorldBench-data/` by default; pass `--dataset <dir>` to use another location. The evaluator
reads `dataset.json` → `catalog.json` and the files the catalog references (reference videos, `hands/*.npz`,
`assets/annotation_*.json`). The format is described in [`docs/inputs.md`](docs/inputs.md) and the dataset card.

## 5. Bind the runtime configuration

`build_runtime_config.py` writes `evaluation/configs/runtime.json` with paths relative to that file, pointing at
`.envs/`, `models/`, the CUDA 12.8 toolkit (used by XLA in the MUSIQ worker) and the CUDA 11 libraries, and
`--check` verifies that every interpreter, weight and video tool exists. Sampling (16 frames), batch sizes and
backend separation are fixed by the template and are part of the benchmark protocol; do not edit them.

```bash
CUDA12_ROOT=${CUDA12_ROOT:-$(.envs/main/bin/python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')}
MUSIQ_LIB_ARGS=()
MUSIQ_CUDA_COMPAT_LIB=${MUSIQ_CUDA_COMPAT_LIB:-"$CUDA12_ROOT/compat/lib.real"}
[ -d "$MUSIQ_CUDA_COMPAT_LIB" ] && MUSIQ_LIB_ARGS+=(--musiq-library-dir "$MUSIQ_CUDA_COMPAT_LIB")
MUSIQ_PY38_LIB=$(dirname "$(dirname "$(UV_PYTHON_INSTALL_DIR=$PWD/.envs/uv-python .envs/main/bin/uv python find 3.8)")")/lib   # Python 3.8 shared library
MUSIQ_LIB_ARGS+=(--musiq-library-dir "$MUSIQ_PY38_LIB")
for lib_dir in .envs/musiq-cuda11/nvidia/*/lib; do MUSIQ_LIB_ARGS+=(--musiq-library-dir "$lib_dir"); done
.envs/main/bin/python evaluation/scripts/build_runtime_config.py \
  --profile release --release-root . \
  --main-cuda-root "$CUDA12_ROOT" --musiq-cuda-root "$CUDA12_ROOT" \
  "${MUSIQ_LIB_ARGS[@]}" --output evaluation/configs/runtime.json --check
```

It prints the path of the written file. `runtime.json` is not shipped with the repository; this step creates it. All
paths inside `runtime.json` are stored **relative to the file itself** (system locations such as the CUDA toolkit
included), so re-run this step after moving the repository to another location; `run_evaluation.sh` passes `--config evaluation/configs/runtime.json` unless
`--config FILE` is given. `prepare` copies the resolved configuration into `runs/<name>/configs/run.json`, and every
later step of that run (workers, `resume`) reads it from there, so editing `runtime.json` does not affect existing
runs. Re-run this step after moving `.envs/` or `models/` separately. `--profile custom` (see `--help`) binds arbitrary locations instead of the
layout above; `--asset KEY=PATH` overrides a single weight (dotted keys such as `sam3.checkpoint`).
[`docs/resources.md`](docs/resources.md) lists every configuration key.

## 6. Check the installation

Unit tests (CPU, ≈ 5 s, 119 tests; they include an end-to-end CLI test of `prepare` on a synthetic dataset and an import test of every backend module):

```bash
cd evaluation && ../.envs/main/bin/python -m pytest -q tests && cd ..
```

Synthetic GPU check **[GPU]** — scores one 13-frame synthetic prediction with MUSIQ (≈ 1 min, writes
`runs/example/reports/scores.json`):

```bash
sh evaluation/scripts/run_evaluation.sh evaluate \
  --dataset evaluation/examples/dataset \
  --predictions evaluation/examples/predictions.json --model Example \
  --metrics musiq --gpus 0 --foreground --run-dir runs/example
```

Reference check **[GPU]** — score the reference videos themselves on six cases (all three embodiments, every metric
family), all 17 metrics:

```bash
echo '["hw500_0014", "hw500_0363", "hw500_0431", "hw500_0050", "hw500_0150", "hw500_0452"]' > cases6.json
sh evaluation/scripts/run_evaluation.sh evaluate --use-reference-videos \
  --cases cases6.json --gpus 0,1 --foreground --run-dir runs/reference6
```

The expected reports of this check, and how closely your numbers should match, are in
[`examples/expected_outputs/`](examples/expected_outputs/README.md).

## 7. Evaluate your model

### Inputs

Generate one video per case from `first_frames/<case_id>.png` and the case prompt (`prompts.jsonl` → `prompt`,
identical to `catalog.json` → `task_description`). Models that take control signals may also use the three control
videos, the 3-D hand joints (`hands/`) and the camera trajectory (`camera/`); never the reference video. Report which
inputs were used and compare only runs with the same inputs (see the
[dataset card](https://huggingface.co/datasets/EmbodiedCity/HandWorldBench#benchmark-protocol)). Name each file `<case_id>.mp4` (`.mov`, `.mkv`, `.avi` also accepted) and
put them in one folder. By default the evaluator assumes the **reference-native protocol**: same width × height as
the reference and the complete clip; if the frame rate or length differs, it maps time uniformly from first to last
decoded frame; if only the size differs, it assumes a full-frame resize. Anything else (crop, pad, undistortion,
retiming) must be declared — see [rectified submissions](examples/rectified_submission/README.md) and
[input formats](docs/inputs.md). An optional `<case_id>.json` next to a video can carry `time_mapping` and
`spatial_mapping` for that case.

### Run **[GPU]**

```bash
sh evaluation/scripts/run_evaluation.sh evaluate \
  --model MyModel --videos generated/MyModel --run-dir runs/MyModel --gpus 0,1
```

* `--model NAME` (required for `--videos` / `--predictions`) labels every result row; one run holds one model.
* `--videos DIR` or `--predictions FILE.json` (a manifest of `{"case_id", "video", optional mappings, optional
  "model"}` rows; relative video paths resolve against the dataset directory).
* `--use-reference-videos` scores the reference videos themselves (an upper-bound / sanity run). The model name
  defaults to `reference`; `--model` overrides it. The name is only a label and does not change any score.
* `--cases cases.json` restricts the run to a JSON list of case ids; `HandWorldBench-data/cases_by_category/{human,handson,real_robot}.json` are ready-made lists.
* `--metrics a,b,c` restricts the metrics (ids in [§10](#10-metrics)); e.g. `pose_control,keypoint_coverage`.
* `--gpus 0`, `0,1`, `1,3,5` or `auto` (indices of the devices visible to the launcher; `CUDA_VISIBLE_DEVICES` is
  respected). Without `--foreground` the scheduler runs in the background and the command returns immediately.
* `--dataset DIR`, `--config FILE` override the defaults (`HandWorldBench-data/`, `evaluation/configs/runtime.json`).

`prepare` instead of `evaluate` only validates inputs and builds the task graph (no GPU work); binding problems are
listed in `runs/MyModel/reports/prepare.json` and become `input_binding_error` slots rather than aborting the run.

A complete submission has one video for each of the 500 cases: `prepare` must report `"videos": 500` and no
`input_errors`. Cases without a video get no result slots, and failed slots are excluded rather than scored 0, so
`summary.json` and `hwbscore.json` cover only the cases actually scored. Report HWBScore only for complete runs, and
check `summary.json` → `common_cases` against `coverage.json`. `report` prints a warning on stderr when a model's
scores cover fewer than 500 cases.

## 8. Monitor, resume and compare runs

```bash
sh evaluation/scripts/run_evaluation.sh status --run-dir runs/MyModel      # task counts per state
sh evaluation/scripts/run_evaluation.sh resume --run-dir runs/MyModel      # restart unfinished / failed tasks
sh evaluation/scripts/run_evaluation.sh report --runs runs/ModelA runs/ModelB --output runs/comparison
```

A run directory is self-contained (`configs/`, `queue.sqlite3`, `results/`, `reports/`, `logs/`). Interrupting a
run (Ctrl-C, node loss) is safe: completed tasks are atomic, and `resume` continues from the queue. Observations are
cached under `cache/` keyed by video fingerprint, model and protocol, so re-running a model — or adding metrics with
`evaluate --metrics ...` on the same run directory — reuses everything already computed. A run lock prevents two
schedulers on one run; GPU locks coordinate concurrent runs that share `cache/`. `report` compares models on their
common cases with matching metric protocols, overall and per embodiment.

## 9. Outputs

```text
runs/MyModel/
  configs/run.json            model name, dataset path, devices, metrics, runtime config, package version
  configs/predictions.json    the bound inputs (video, SHA-256, time and spatial mapping per case)
  configs/tasks.json          observation and scoring task graph
  configs/slots.json          one slot per (case, metric) with its applicability
  queue.sqlite3               task states; what status / resume read
  results/<task_id>/          per-task results
  reports/prepare.json        binding summary and input errors
  reports/scores.json         one record per (case, metric)
  reports/summary.json        per-metric means on common cases, overall and per embodiment
  reports/hwbscore.json       six dimension means and the 0-100 HWBScore
  reports/live_status.json, reports/monitor.jsonl   progress and GPU occupancy
  logs/scheduler.log, logs/events.jsonl, logs/<worker>.log
```

A `scores.json` record has `case_id`, `model`, `category` (`human`, `handson`, `real_robot`), `metric_id`,
`metric_name`, `metric_protocol`, `applicability`, `status`, `raw_score`, `score_0_1` and failure details. `status` is
`ok`, `scored_with_missing_penalty` (some expected points / units missing in the prediction and penalised) or
`partial_reference_population` (only part of the reference population is available) for a scored slot. `observed`
marks per-case JEPA features (JEPA Similarity is scored per collection in `summary.json`). `not_applicable`,
`missing_reference` (also `missing_reference_entity_region`, `missing_reference_object_geometry`),
`input_binding_error`, `execution_error` and `pending` keep a null score and are never averaged as zero. Use
`score_0_1 is not None` rather than `status == "ok"` to select scored slots. Missing predictions inside a valid input (e.g. a hand the model
failed to draw) receive the metric's penalty and are scored.

`hwbscore.json` averages each of the six dimensions over its metrics' per-metric means (computed over the cases
scored for every model in the report, `common_cases`) and reports the equally weighted mean × 100.
Metrics that are not applicable to a whole scope are excluded; an incomplete metric selection, mixed protocols or an
empty dimension leaves the total `null`.

## 10. Metrics

| Category | Metric | ID |
|---|---|---|
| Visual Quality | MUSIQ | `musiq` |
| Visual Quality | Visual Plausibility | `visual_plausibility` |
| Visual Quality | JEPA Similarity | `jepa_similarity` |
| Hand Fidelity | Pose Control | `pose_control` |
| Hand Fidelity | Keypoint Coverage | `keypoint_coverage` |
| Hand Fidelity | Hand Morphology | `hand_morphology` |
| Appearance Consistency | Entity Consistency | `entity_consistency` |
| Appearance Consistency | Background Consistency | `background_consistency` |
| Appearance Consistency | Photometric Consistency | `photometric_consistency` |
| Instruction Following | Action Adherence | `action_adherence` |
| Instruction Following | Event Adherence | `event_adherence` |
| Physical Interaction | Trajectory Accuracy | `trajectory_accuracy` |
| Physical Interaction | Causal Plausibility | `causal_plausibility` |
| Physical Interaction | Evolution Correctness | `evolution_correctness` |
| Physical Interaction | State Regulation | `state_regulation` |
| Temporal Coherence | State Progression | `state_progression` |
| Temporal Coherence | Memory Consistency | `memory_consistency` |

### Hand observations

Grounding DINO tiny detects hands with box/text thresholds 0.25 and NMS IoU 0.5. Human inputs use `hand.`; robotic inputs use `robotic hand. mechanical hand.`. RTMPose-m-Hand5 predicts 21 points from each detected hand crop using BGR pixels, 256 × 256 input, and padding 1.25. Prediction confidence threshold is 0.3. SAM3 consumes the same hand detections for entity masks. Hand localization and predicted keypoints use these models directly.

Morphology, state, and memory localization consume the same DINO detection protocol. Pixel-exact receipts are shared across consumers and isolated by model, prompt, video fingerprint, and reference/prediction role. State localization selects the two highest-confidence deduplicated instances without assigning anatomical sides. With no object prompts, SAM3 loading and image encoding are skipped. Morphology batches crops across frames (`batch.hand_morphology`, default 8), restores frame/instance bindings, and preserves missing-hand penalties and frame aggregation. OOM reduces the crop batch size.

The default sampling is 16 frames. Reference points are read from the dataset's geometry and transformed using the video's declared temporal and spatial mappings. Both keypoint metrics evaluate **all reference-valid, in-frame points, including occluded points**. No visibility classifier is involved.

Pose Control assigns each point:

```text
1 - min(pixel_distance / (0.1 * max(image_width, image_height)), 1)
```

Missing or low-confidence predicted points receive 0. Point scores are averaged within each reference-valid frame, then across frames. Keypoint Coverage is the number of detected reference points divided by the fixed number of reference-valid, in-frame points. Hand identity, confidence, finite coordinates, and image bounds determine detection validity.

Entity Consistency includes hand instances with at least one valid reference point inside the image during the sampled interval. A hand that stays outside the image is outside the required entity population. Prediction misses for an eligible reference entity retain their zero-score penalty.

Entity Consistency and Trajectory Accuracy compare the objects found in the generated video with a fixed set of reference objects shipped with the benchmark (`reference_objects/`: object list, masks and geometry on the 16 sampled reference frames). The reference side is therefore not re-observed: no objects are selected, localised or tracked on the reference video, and every run uses the same reference objects. The object-localisation model and SAM 3 run only on the generated video. A custom dataset without `reference_objects` falls back to selecting objects on its reference video.

### Score rules

| Metric | Published score |
|---|---|
| MUSIQ | `clip(raw / 100, 0, 1)` |
| JEPA Similarity | `exp(-0.4 * max(distance, 0))` from collection-level JEPA features |
| Hand Morphology | Mean normal-hand probability with missing expected hands scored 0 |
| Photometric Consistency | Equal mean of first/previous branches, each using `1 / (1 + MSE / 0.01)` on normalized RGB |
| Causal Plausibility | Fixed 0–3 rating divided by 3 |
| Other metrics | Their fixed positive 0–1 scales |

Trajectory Accuracy interpolates a single internal missing center from its two original neighboring observations and actual sample times. Consecutive and endpoint gaps receive the prescribed missing penalty. Entity observations and semantic scoring use their reference-defined evaluation units. Semantic unknown answers receive 0. Execution failures, missing reference data, pending annotations, and inapplicability retain distinct statuses and null scores.

Event and memory annotations affect their respective evidence and scores. Unchanged frame, detection, keypoint, mask, and feature observations remain reusable. Hand Morphology applies to human hands. JEPA Similarity is aggregated over a collection of video features.

### VLM judges

All judge prompts and judge-read annotations are in English. The judges decode greedily (`do_sample=False`), except the numeric state answers, which sample with a fixed seed
(`torch.manual_seed(1729)`, temperature 1.0, top-p 0.95, top-k 20). Scores are therefore deterministic for a fixed software and
hardware stack; small numeric differences between GPU types or library versions can flip individual judge answers.

## 11. Runtime and GPU memory

Measured on NVIDIA H20 (96 GB) GPUs, weights on a local RAM disk, without `causal-conv1d`:

| Run | Wall-clock | Notes |
| --- | --- | --- |
| synthetic MUSIQ check (§6) | ≈ 1 min | one GPU, one 13-frame clip |
| reference check (§6): 6 cases, all 17 metrics, two GPUs (`--gpus 0,1`) | 31 min | 201 tasks, model loading included; most of the time is state evidence; reports in [`examples/expected_outputs/`](examples/expected_outputs/README.md) |

Compute grows roughly linearly with the number of cases; models stay resident between tasks, so the per-case cost
of a full 500-case run is lower than in the 6-case run. More GPUs (`--gpus 0,1,...`) run independent tasks in
parallel. The scheduler budgets GPU memory per backend; the largest is the visual-plausibility model (≈ 76 GB), the
state / memory judge with SAM 3 uses ≈ 35 GB, the others less. Cache: ≈ 120 MB per case (`cache/`).

Score stability:

* VLM-judge metrics (Visual Plausibility, Action / Event Adherence, Causal Plausibility, Evolution Correctness,
  State Regulation / Progression, Memory Consistency) are deterministic for a fixed software and hardware stack, but
  another GPU type or other library versions can change an individual judge answer near a decision boundary.
* The hand-based metrics (Pose Control, Keypoint Coverage, Hand Morphology) depend slightly on which other metrics run
  in the same job: hand detections are batched and shared across consumers, so the batch composition changes the
  last digits (typically 1e-5 to 5e-4). Compare models with the same metric selection.
* A task that crashes is reported as `execution_error` (never as 0); `resume` retries it (a deterministic crash fails
  again; see §12).

## 12. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Create .envs/main in the repository root ...` | the launcher looks for `.envs/main/bin/python` in `evaluation/` and its parents; create the environments in the repository root (step 2) |
| `venv` fails with `Operation not supported` on `lib64` | the filesystem has no symbolic links (e.g. ossfs); create `.envs/` on a local disk and symlink it into the repository |
| `build_runtime_config.py --check` reports `Missing model asset ...` | a weight is not at the path of step 3; fix it or pass `--asset KEY=PATH` |
| `New runs require --dataset, --model, and an input mode` | give `--model NAME` with `--videos` / `--predictions` (it is optional only with `--use-reference-videos`) |
| `Unknown video case ID: ...` | a file in `--videos` is not named `<case_id>.mp4`; remove or rename it |
| `input_binding_error` slots in `reports/prepare.json` | the video or its declared mapping could not be bound to the reference (the error text in `prepare.json` names the reason). A size or length difference alone is mapped automatically (§7); crops, pads, undistortion and retiming must be declared — see [rectified submissions](examples/rectified_submission/README.md) |
| `Cannot modify an active run` / `A selected GPU is reserved by another HandWorldBench run` | a scheduler is still running on this run or on the same GPUs and cache; wait, or check with `status` |
| CUDA out of memory | the visual-plausibility model needs ≈ 76 GB on one GPU; other GPU jobs on the same card can push it over; choose an idle GPU with `--gpus` |
| MUSIQ fails to load `libcudnn` / `libcublas` | the CUDA 11 libraries are not in the MUSIQ `LD_LIBRARY_PATH`; re-run step 5 with the `--musiq-library-dir` loop |
| warnings `causal_conv1d ... falling back to its reference PyTorch implementation` | harmless; install `causal-conv1d` (step 2) for speed |
| `execution_error` on Pose Control / Keypoint Coverage; the hand-keypoint worker exited with code `-11` (kernel log: segfault in `libnvidia-ptxjitcompiler`); persists after `resume` | the ONNX Runtime keypoint session crashed in the driver's PTX JIT because it was the first CUDA user of a fresh worker process. The evaluator therefore creates a PyTorch CUDA context first (`handworldbench/observations/hands.py`, `KeypointEstimator`). If it still occurs with another driver / ONNX Runtime, re-run the affected cases into a new `--run-dir` with a fresh `cache/` (e.g. `--cases failed.json --metrics pose_control,keypoint_coverage`) and compare models on the same metric selection |
| `report` warns that scores cover fewer than 500 cases | the run is partial (missing videos or failed slots); HWBScore averages only the scored cases — complete the run before comparing models (§7) |

Unit tests: `cd evaluation && ../.envs/main/bin/python -m pytest -q tests`. Input format details:
[docs/inputs.md](docs/inputs.md). Configuration keys and storage: [docs/resources.md](docs/resources.md).
