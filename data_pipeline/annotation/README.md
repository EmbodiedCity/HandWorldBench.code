# Captions and operation labels

`annotate.py` writes, for every screened clip, an English **global caption**, one **segment caption** per contiguous
segment, and one dominant **operation label** from the seven-family / 22-sub-type taxonomy in `taxonomy.json`. It
calls a vision-language model served by [vLLM](https://github.com/vllm-project/vllm) through its OpenAI-compatible
API; `merge.py` then validates the results and joins them to the screening manifest.

| Output | Frames shown | Length | Token budget |
| --- | --- | --- | --- |
| global caption | 5, spread over the whole clip | one or two short paragraphs (≤ ~90 words) | 4,096 |
| segment captions | 5 per segment, `max(3, ceil(frames / 240))` segments | two paragraphs each (≤ ~120 words) | 8,192 each |
| operation label | the 5 global frames + the taxonomy + the new global caption + up to 4 segment captions | JSON: primary code or `UNKNOWN`, evidence frames, rationale | 4,096 |

The prompts are in `legacy_prompts.py` (caption) and `annotate.py` (classification). The taxonomy has seven families,
22 base sub-types plus `M5` Push / Roll, and `UNKNOWN` for unresolved clips. Each operation and its evidence frames are
validated independently of the caption.

## Installation

CPU side (from the repository root; the same environment as screening):

```bash
python -m venv .venv-pipeline && . .venv-pipeline/bin/activate
python -m pip install -r data_pipeline/annotation/requirements.txt
```

Model side **[GPU]**: install vLLM in its own environment and serve a Qwen vision-language checkpoint that supports
image messages and the `enable_thinking` chat-template switch, with a context long enough for five images plus the
output budget, at a loopback address under an explicit alias. For example:

```bash
vllm serve <your-qwen-vl-checkpoint> --served-model-name hwb-vlm --host 127.0.0.1 --port 18471
```

The checkpoint and vLLM are not part of this repository; use them under their own licences.

## Quick check (CPU)

```bash
cd data_pipeline/annotation && python -B test_contracts.py && cd ../..
python data_pipeline/annotation/examples/make_input.py --out runs/synthetic-caption-input
```

The tests (6, < 1 s) decode an FFmpeg-generated clip and use an in-process model substitute to check the requests,
frame sampling, output structure and operation validation. `make_input.py` writes an 81-frame test pattern
(`pattern.mp4`) and a one-line manifest you can feed to `annotate.py`.

## Usage **[GPU]**

The input is a screening manifest (JSONL) with `id`, `domain` (`ego`, `handson` or `ti5`), `frames`,
`source_frames`, `video` and `video_sha`; paths are relative to the manifest. An optional `videos` map carries export
receipts (`path`, `sha256`, `decoded_frames`) per view.

```bash
python data_pipeline/annotation/bind_service.py \
  --endpoint http://127.0.0.1:18471 --model hwb-vlm --output runs/vlm-service.json

python data_pipeline/annotation/annotate.py \
  --input runs/synthetic-caption-input/manifest.jsonl --out runs/captions \
  --model hwb-vlm --service-manifest runs/vlm-service.json

python data_pipeline/annotation/merge.py \
  --input runs/synthetic-caption-input/manifest.jsonl --run-dir runs/captions \
  --output runs/annotated.jsonl
```

* `bind_service.py` records the served model name and vLLM version; refresh it after redeploying, and use a new
  `--out` directory when the model or service changes.
* `annotate.py` writes one validated sidecar per clip (video SHA, frame mapping, captions, operation, provenance) under
  `--out` and a journal `results.jsonl`. Re-running the same command resumes: units are reused when service, code, prompts, taxonomy,
  source range and video content all match. `--help` lists sampling, concurrency and thinking options.
* `merge.py` checks the input SHA, ids, sidecars and signatures and writes a new manifest that keeps the screening
  fields and adds `caption_path`, `caption_sha256`, `task_text`, `segments`, `operation` and `operation_provenance`
  (including a `human_review` field for the human check that follows). It refuses to overwrite an existing `--output`.

### Re-captioning clips that already have operation labels

`reuse_operations.py` binds accepted labels from an earlier annotation manifest (matching id, domain, video SHA and
source range); `--preserve-operation` then rewrites the captions and keeps those labels with their provenance:

```bash
python data_pipeline/annotation/reuse_operations.py \
  --input runs/handson/manifest.native_distribution.jsonl \
  --previous inputs/handson-previous-annotations.jsonl --output runs/handson-recaption.jsonl

python data_pipeline/annotation/annotate.py --preserve-operation \
  --input runs/handson-recaption.jsonl --out runs/handson-captions \
  --model hwb-vlm --service-manifest runs/vlm-service.json
```

## Settings used for the released datasets

* Every request is sent with `temperature=0.2` and `chat_template_kwargs={"enable_thinking": true}`; the served model
  must accept that switch (Qwen chat templates do). Outputs are therefore not bit-reproducible across runs.
* The released captions and labels were produced with `Qwen3.8-27B`, served locally with vLLM.
* The classifier returns one dominant code or `UNKNOWN`. In the released corpus, clips left `UNKNOWN` carry the code
  `OTHER` with `family: null` (156 clips).
* Every output is validated automatically (schema, frame coverage, operation validation in `caption_contract.py`).
  All released captions and labels were verified by human annotators.

## Files

| File | Role |
| --- | --- |
| `annotate.py` | caption and classification runner |
| `legacy_prompts.py` | caption prompts (frozen; their hash is part of the resume key) |
| `taxonomy.json` | operation families, sub-types and definitions |
| `caption_contract.py` | output validation |
| `bind_service.py`, `manifest_io.py`, `merge.py`, `reuse_operations.py` | service receipt, manifest I/O, merge, label reuse |
| `test_contracts.py`, `examples/make_input.py` | tests and synthetic input |

Previous stage: [screening](../screening/README.md).
