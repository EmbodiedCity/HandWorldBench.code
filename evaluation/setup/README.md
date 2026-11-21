# Environment files

The installation steps are in the [evaluator guide](../README.md#2-install-the-environments). This directory holds
the pinned requirement files and the model manifest they refer to.

| File | Installed into | Notes |
| --- | --- | --- |
| `requirements.gpu-common.txt` | included by the three files below | NumPy, SciPy, Pillow, OpenCV, einops, tqdm pinned over the NVIDIA PyTorch 25.03 base |
| `requirements.main.txt` | `.envs/main` | scheduler, hand detection / keypoints, judges (Transformers 5.20.2, ONNX Runtime GPU 1.20.2) |
| `requirements.sam3.txt` | `.envs/sam3` | SAM 3 segmentation and background workers (timm 1.0.22) |
| `requirements.metrics.txt` | `.envs/metrics` | JEPA and hand-morphology workers (timm 1.0.9); add `vjepa==0.1.2` with `--no-deps` |
| `requirements.musiq.txt` | `.envs/musiq` (Python 3.8) | JAX 0.4.1 + CUDA 11 wheel, TensorFlow 2.7, Flax 0.3.3 |
| `requirements.musiq-cuda11.txt` | `.envs/musiq-cuda11` (`pip install --target`) | CUDA 11 / cuDNN 8 runtime libraries for MUSIQ |
| `model_resources.json` | — | every weight: path under `models/`, upstream URL, pinned revision, SHA-256 |

Recorded versions of the reference environment: PyTorch `2.7.0a0+7c8ec84dab.nv25.3`, torchvision `0.22.0a0`,
Transformers `5.20.2`, ONNX Runtime GPU `1.20.2`, NumPy / SciPy `1.26.4` / `1.15.2` (main), decord `0.6.0`; MUSIQ:
JAX / jaxlib `0.4.1` / `0.4.1+cuda11.cudnn86`, TensorFlow `2.7.0`, Flax `0.3.3`, NumPy `1.21.6`.
