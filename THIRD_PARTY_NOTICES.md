# Third-party notices

HandWorldBench's own code is released under the MIT licence in [`LICENSE`](LICENSE). The components below are
third-party. Third-party components vendored under `evaluation/handworldbench/_vendor/` are distributed under their own
licences, which are included in their directories; the licence text of each vendored component is shipped next to it.
Model weights are **not** redistributed: the evaluator downloads them from their upstream hosts, and each keeps the
terms of its upstream release.

## Vendored source code

Paths are relative to `evaluation/handworldbench/_vendor/engine/engines/`.

| Component | Path | Upstream | Revision | Licence (file) | Used for |
| --- | --- | --- | --- | --- | --- |
| SAM 3 | `classic/metric_suite/sources/SAM3/` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | `0570b3a` | SAM License (`LICENSE`) | hand / object / background masks |
| SAM 2 (code adapted inside SAM 3) | `classic/metric_suite/sources/SAM2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | as included in SAM 3 | Apache-2.0 (`LICENSE`) | notice only: SAM 3 contains code derived from SAM 2 |
| Detectron2 (code adapted inside SAM 3) | `classic/metric_suite/sources/Detectron2/` | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) | as included in SAM 3 | Apache-2.0 (`LICENSE`) | notice only: SAM 3 contains code derived from Detectron2 |
| DINOv3 | `classic/metric_suite/sources/DINOv3/` | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | `6876159` | DINOv3 License (`LICENSE.md`) | Background Consistency features |
| OpenAI CLIP | `classic/metric_suite/sources/OpenAI-CLIP/` | [openai/CLIP](https://github.com/openai/CLIP) | `d05afc4` | MIT (`LICENSE`) | Entity Consistency features |
| VBench | `classic/metric_suite/sources/VBench/` | [Vchitect/VBench](https://github.com/Vchitect/VBench) | `fd18b3d` | Apache-2.0 (`LICENSE`) | shared helpers |
| VBench-2.0 hand anomaly detector | `classic/metric_suite/sources/VBench2-Hand/` | [Vchitect/VBench](https://github.com/Vchitect/VBench) (VBench-2.0) | `fd18b3d` | Apache-2.0 (`LICENSE`, copied from the upstream repository) | Hand Morphology |
| JEDi | `classic/metric_suite/sources/JEDi/` | [oooolga/JEDi](https://github.com/oooolga/JEDi) | `e64f3fe` | Apache-2.0 (`LICENSE.txt`) | JEPA Similarity distance |
| WorldArena (one helper, `video_quality/JEDi/batch.py`) | `classic/metric_suite/sources/WorldArena/` | [tsinghua-fib-lab/WorldArena](https://github.com/tsinghua-fib-lab/WorldArena) | `2da2ae2` | no licence file upstream; included with the permission of its authors, who are part of the HandWorldBench team | batching for JEPA features |
| MUSIQ | `classic/MUSIQ/musiq/` | [google-research/google-research/musiq](https://github.com/google-research/google-research/tree/master/musiq) | not recorded | Apache-2.0 (`classic/MUSIQ/LICENSE`, from the upstream repository) | MUSIQ |
| Depth Anything 3 | `new_aux/vendor/Depth-Anything-3/` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | not recorded | Apache-2.0 (`LICENSE`, from the upstream repository) | Photometric Consistency depth / pose |
| WBench (three metric modules) | `new_aux/vendor/WBench/` | [meituan-longcat/WBench](https://github.com/meituan-longcat/WBench) | not recorded | MIT (`LICENSE`, from the upstream repository) | Visual Plausibility, Causal Plausibility, VLM judge helpers |

The licence files of WBench, Depth Anything 3, MUSIQ and VBench-2.0 are copied from the upstream repositories.

One first-party file contains adapted third-party code (path relative to the repository root):

| Component | Path | Upstream | Revision | Licence (file) | Used for |
| --- | --- | --- | --- | --- | --- |
| RTMPose SimCC post-processing | `evaluation/handworldbench/observations/pose_geometry.py` | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | not recorded | Apache-2.0 (`evaluation/handworldbench/observations/LICENSE-mmpose.txt`) | Pose Control / Keypoint Coverage keypoint decoding |

## Model weights (downloaded by the user, not redistributed)

| Asset | Upstream | Access | Terms (as published upstream; check the model card before commercial use) |
| --- | --- | --- | --- |
| Grounding DINO tiny | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | open | Apache-2.0 |
| RTMPose-m Hand5 (ONNX) | [OpenMMLab download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip) | open | Apache-2.0 (MMPose) |
| Qwen3-VL-4B-Instruct | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | open | see model card (Apache-2.0 at the pinned revision) |
| Qwen3.5-9B (judge) | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | open | see model card |
| SAM 3 checkpoint | [facebook/sam3](https://huggingface.co/facebook/sam3) | **gated** (request access) | SAM License |
| OpenAI CLIP ViT-B/32 | [OpenAI](https://github.com/openai/CLIP) | open | MIT |
| DINOv3 ViT-S/16 LVD-1689M | [Meta request form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | **gated** (request access) | DINOv3 License |
| MUSIQ SPAQ checkpoint | [Google Research](https://storage.googleapis.com/gresearch/musiq/spaq_ckpt.npz) | open | see upstream repository |
| Visual-plausibility model, DA3-GIANT-1.1 | [meituan-longcat/WBench-weights](https://huggingface.co/meituan-longcat/WBench-weights) | open | see model card; Depth Anything 3 giant weights are published for **non-commercial** use upstream |
| VBench-2.0 hand anomaly detector | [Vchitect/VBench-2.0_models](https://huggingface.co/Vchitect/VBench-2.0_models) | open | see model card |
| V-JEPA ViT-H/16 and SSv2 probe | [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | open | CC BY-NC 4.0 (**non-commercial**) |

## Screening

`data_pipeline/screening/handclip/aesthetic_service.py` ranks Ego clips with OpenAI CLIP ViT-L/14 (MIT) and the LAION
improved aesthetic predictor ([christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor),
Apache-2.0); the weights are downloaded by the user and not redistributed.

## Assets for control-signal rendering

| Asset | Source | Version | Terms | In this repository |
| --- | --- | --- | --- | --- |
| MANO hand model (`MANO_LEFT.pkl`, `MANO_RIGHT.pkl`) | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) | v1.2 | registration required; MANO licence (non-commercial research) | no; download it yourself |
| Wuji hand description (the HandsOn simulated hand) | [wuji-technology/wuji-hand-description](https://github.com/wuji-technology/wuji-hand-description) | commit `bbc2f64eec834aae616eacc2db1f69016f860c2d` | MIT | the landmark files `data_pipeline/controls/assets/{left,right}_landmarks.xml` (the upstream MJCF with the 21 HandWorldBench landmark sites added; MIT) are shipped; the link meshes are cloned from upstream |
| smplx (Python package, MANO layer) | [vchoutas/smplx](https://github.com/vchoutas/smplx) ([PyPI](https://pypi.org/project/smplx/)) | 0.1.28 | SMPL-X / MPI licence (non-commercial research) | no; installed with pip |
| chumpy (Python package, reads the MANO `.pkl` files) | [mattloper/chumpy](https://github.com/mattloper/chumpy) ([PyPI](https://pypi.org/project/chumpy/)) | 0.70 | MIT | no; installed with pip |

Installation steps: [`data_pipeline/controls/README.md`](data_pipeline/controls/README.md).

Some of the weights above are gated or published for non-commercial use only. Using the evaluator therefore
inherits those terms, regardless of the MIT licence of the HandWorldBench code.
