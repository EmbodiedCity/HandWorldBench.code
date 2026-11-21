# Runtime resources

Install the backend environments as in the [evaluator guide](../README.md#2-install-the-environments). The `scripts/build_runtime_config.py` command binds local resources to `configs/runtime.example.json` and writes a runtime file for `--config`. Runtime paths are relative to that file's directory. If the dataset embeds its own `runtime` object, those paths are relative to `dataset.json`. A previously saved runtime can continue using its established absolute paths.

The generator accepts an optional `--asset KEY=PATH` for a weight or model directory stored elsewhere. Nested model keys such as `sam3.checkpoint` and `jepa.encoder` use dots. The paths passed on the command line are interpreted from the working directory and rewritten relative to the generated config. Keep the configured resources and evaluator in the same relative layout when moving them.

## Model assets

| Configuration key | Asset | Consumers |
|---|---|---|
| `hand_detector` | Grounding DINO tiny Hugging Face model directory | Shared hand detection |
| `keypoints` | RTMPose-m-Hand5 ONNX | Pose Control, Keypoint Coverage |
| `sam3` | Source directory, checkpoint, BPE vocabulary | Hand/object masks and background/state evidence |
| `vision` | Qwen3-VL-4B-Instruct | Manipulated object localization |
| `clip` | OpenAI CLIP ViT-B/32 checkpoint | Entity Consistency |
| `dinov3` | DINOv3 ViT-S/16 LVD checkpoint | Background Consistency |
| `musiq` | MUSIQ SPAQ checkpoint | MUSIQ |
| `pavrm` | Visual plausibility reward model directory | Visual Plausibility |
| `jepa.encoder` / `jepa.probe` | V-JEPA ViT-H/16 and Something-Something-v2 probe | JEPA Similarity |
| `hand_morphology` | VBench2-Hand classifier checkpoint | Hand Morphology |
| `judge` | Qwen3.5-9B directory | Action, event, and causal scoring |
| `state_judge` | Qwen3.5-9B directory | State and memory scoring |
| `da3` | Depth Anything 3 Giant model directory | Photometric Consistency |

The [model resource manifest](../setup/model_resources.json) lists expected paths within `models/`, upstream sources, and the reference checksums available for particular files. For a model directory, a config checksum identifies only `config.json`; obtain the complete snapshot with its tokenizer, processor, configuration and weight shards. The `keypoints` entry expects `end2end.onnx` extracted from the listed archive. DINOv3 uses the upstream PyTorch `.pth` checkpoint. The SAM3 vocabulary is bound to the existing vendored CLIP copy of the same file. Third-party implementation code is under `handworldbench/_vendor/engine`, and model weights are required as separate runtime inputs. The default runtime sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for worker processes. Obtain the model files during setup before starting evaluation.

## Environments

`default_python` runs ordinary GPU workers. The `python` mapping can select a different interpreter for `musiq`, `segmentation`, `background`, `anatomy`, `pavrm`, `jepa`, `judge`, `state`, `hands`, or `regions`. The scheduler keeps these processes resident while compatible tasks remain.

The deployed environment uses Python 3.12 for the main backends and Python 3.8 for MUSIQ. Its recorded package versions are:

| Component | Recorded version |
|---|---|
| Main PyTorch | `2.7.0a0+7c8ec84dab.nv25.3` |
| Main torchvision | `0.22.0a0` |
| Transformers | `5.20.2` |
| ONNX Runtime GPU | `1.20.2` |
| NumPy / SciPy, main | `1.26.4` / `1.15.2` |
| timm, main / SAM3 / JEPA and hand morphology | `1.0.30` / `1.0.22` / `1.0.9` |
| decord | `0.6.0` |
| MUSIQ JAX / jaxlib | `0.4.1` / `0.4.1+cuda11.cudnn86` |
| MUSIQ TensorFlow / Flax | `2.7.0` / `0.3.3` |
| MUSIQ NumPy | `1.21.6` |

Backend environments also provide OpenCV, Pillow, their model-specific dependencies, and matching CUDA libraries. Install backend dependencies in the dedicated environments. Existing installations with extra dependency directories can bind them via `--main-python-path` and `--musiq-python-path`. Runtime outputs record the actual ONNX providers, JAX devices, and inference batches.

## Storage and environment variables

- `cache_root`: local staged videos, decoded frames, model observations, and device locks. Share this root across models to reuse compatible reference observations.
- `models_root`: local root for vision-language model directories.
- `classic_data`: root for the classic model resource layout. The generator defaults to `models/classic` and accepts `--classic-data` to select another layout. Create this directory to use the default layout. The runtime explicitly binds the active checkpoints listed in the manifest.
- `env`: subprocess environment variables such as CUDA library paths, CPU thread counts, and model cache locations.
- `backend_env`: environment overrides for individual backends. MUSIQ's CUDA 11 build uses cuDNN 8 and CUDA 11 cuBLAS/runtime libraries. Bind these with `--musiq-library-dir` and set the toolkit via `--musiq-cuda-root`. The other workers use the CUDA 12 environment.
- `HANDWORLDBENCH_CLASSIC_DATA_ROOT` and `HANDWORLDBENCH_MODEL_ROOT`: resource roots for bundled backend modules.

The installation installs the main package in editable mode (`pip install -e evaluation`). At execution the scheduler makes the running HandWorldBench source available to backend subprocesses.

## Scheduling controls

`cpu_workers` sets the decoding/aggregation worker count. `backend_resources` declares a backend's device count, expected memory per device, and optional model load time. These values guide placement and do not impose an allocator memory quota.

```json
{
  "cpu_workers": 4,
  "backend_resources": {
    "pavrm": {"gpu_count": 1, "memory_mib_per_gpu": 76000}
  },
  "batch": {"hand_detector": 8, "keypoints": 32},
  "sampling": {"frames": 16}
}
```

`resource_wait_timeout_s` controls how long an otherwise idle queue waits for sufficient GPU resources before recording the affected task as an execution failure. The default is 120 seconds. Remaining feasible tasks continue.

## Inference options

State and memory evidence collect hand-detection images into bounded batches
using `batch.hand_detector` (default 8). State context frames and subsequent
detail crops are batched separately because the detail crop depends on the
context detections. Memory frames shared by multiple windows are inferred once.
Object segmentation keeps its per-image prompt state. Detection timing records
include actual inference batches, cache hits, and OOM reductions.

`observation_options` applies settings to one observation family. For example,
`"memory_evidence": {"thinking_tokens": 1024}` changes only the memory judge's
numeric decision budget. A value of `0` uses direct finite-label classification.
Omitting the setting preserves the model call's default budget. The setting is
included in the observation cache key and is reset between worker tasks.
