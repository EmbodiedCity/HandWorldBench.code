"""Evaluate images with the official Google Research MUSIQ implementation.

This is an evaluation wrapper, not a reimplementation of MUSIQ.  The model,
checkpoint loading, and image preprocessing are taken from the official
``musiq.run_predict_image`` module.

The first version intentionally evaluates images (or an image directory).
When video data is available, a video adapter can extract frames and call
``MusiqScorer.score_image`` without changing the scoring code.
"""

from __future__ import annotations

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load

# Resolve metric engine modules.
import sys as _final_sys
from pathlib import Path as _FinalPath
_final_engine = next(p for p in _FinalPath(__file__).resolve().parents if (p/'runtime_paths.py').is_file())
if str(_final_engine) not in _final_sys.path: _final_sys.path.insert(0,str(_final_engine))
from runtime_paths import FINAL_ENGINE, FINAL_ROOT, FINAL_DATA, FINAL_RESULTS, FINAL_SUITE

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official MUSIQ inference on one image or an image directory. "
            "The output contains one record per image and a separate summary."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="An image file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path. A sibling .summary.json is also written.",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=[],
        help=(
            "Path to an official MUSIQ .npz checkpoint. Repeat this option "
            "for multiple checkpoints."
        ),
    )
    parser.add_argument(
        "--all-four",
        action="store_true",
        help=(
            "Evaluate the four final quality checkpoints in the standard "
            "checkpoint directory: AVA, KonIQ, PaQ2PiQ and SPAQ."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(FINAL_DATA/'musiq'/'checkpoints'),
        help="Directory used by --all-four.",
    )
    parser.add_argument(
        "--musiq-source",
        default=_hb_path('engines/classic/MUSIQ'),
        help=(
            "Directory containing the official source package. "
            "Default: engines/classic/MUSIQ"
        ),
    )
    parser.add_argument(
        "--num-classes",
        choices=("auto", "1", "10"),
        default="auto",
        help=(
            "Output setting for a single checkpoint. auto uses 10 for AVA "
            "and 1 for the other final checkpoints. Default: auto."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover images below --input when it is a directory.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a small validation run.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Set JAX to CPU before importing MUSIQ. Useful for a shared server.",
    )
    return parser.parse_args()


def discover_images(input_path: Path, recursive: bool) -> List[Path]:
    """Returns a deterministic list of supported image paths."""
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path.resolve()]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        p.resolve()
        for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found under: {input_path}")
    return images


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Linear-interpolated percentile without requiring NumPy in the wrapper."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def aggregate_scores(scores: Sequence[float]) -> Dict[str, Optional[float]]:
    """Aggregates frame/image scores for later reuse by a video adapter."""
    values = [float(value) for value in scores if math.isfinite(float(value))]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


class MusiqScorer:
    """Cached MUSIQ model using the official implementation."""

    def __init__(self, musiq_source: Path, checkpoint: Path, num_classes: int):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if not musiq_source.is_dir():
            raise FileNotFoundError(f"MUSIQ source directory does not exist: {musiq_source}")

        # The official source is not installed as a system package.  Adding its
        # root to sys.path keeps the wrapper independent of the current folder.
        source_text = str(musiq_source.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        try:
            import jax
            import jax.numpy as jnp
            import numpy as np
            import musiq.run_predict_image as official
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Cannot import MUSIQ. Activate the configured MUSIQ Python 3.8 environment "
                "and make sure --musiq-source points to the official MUSIQ source directory."
            ) from exc

        self._jax = jax
        self._jnp = jnp
        self._np = np
        self._official = official
        self.checkpoint = checkpoint.resolve()
        self.num_classes = num_classes

        # These configs and parameters come directly from the official loader.
        self.model_config, self.pp_config, self.params = (
            official.get_params_and_config(str(self.checkpoint))
        )
        self.model = official.model_mod.Model.partial(
            num_classes=self.num_classes,
            train=False,
            **self.model_config,
        )
        self._score_values = jnp.arange(
            1, self.num_classes + 1, dtype=np.float32
        )

    def score_image(self, image_path: Path) -> float:
        """Returns one official MUSIQ score for an image."""
        image = self._official.prepare_image(str(image_path), self.pp_config)
        logits = self.model.call(self.params, image)
        predictions = logits
        if self.num_classes > 1:
            predictions = self._jax.nn.softmax(logits)
        score = self._jnp.sum(predictions * self._score_values, axis=-1)
        return float(self._np.asarray(score)[0])


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Writes a convenient CSV copy alongside JSONL when requested by users."""
    if not records:
        return
    fieldnames = ["image", "score", "status", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
                writer.writerow({key: record.get(key) for key in fieldnames})


FINAL_CHECKPOINTS = (
    ("ava", "ava_ckpt.npz", 10),
    ("koniq", "koniq_ckpt.npz", 1),
    ("paq2piq", "paq2piq_ckpt.npz", 1),
    ("spaq", "spaq_ckpt.npz", 1),
)


def checkpoint_name(checkpoint: Path) -> str:
    """Creates a stable short name for output files."""
    name = checkpoint.stem.lower()
    if name.endswith("_ckpt"):
        name = name[:-5]
    return name.replace("-", "_")


def output_path_for_checkpoint(
    requested_output: Path, checkpoint_name_value: str, multiple: bool
) -> Path:
    if not multiple:
        return requested_output
    return requested_output.with_name(
        f"{requested_output.stem}_{checkpoint_name_value}{requested_output.suffix}"
    )


def evaluate_one_checkpoint(
    *,
    images: Sequence[Path],
    input_path: Path,
    output_path: Path,
    checkpoint: Path,
    num_classes: int,
    musiq_source: Path,
) -> Dict[str, Any]:
    print(f"\n===== {checkpoint_name(checkpoint)} =====")
    print(f"Images found: {len(images)}")
    print(f"Loading weights: {checkpoint}")
    print(f"Number of classes: {num_classes}")
    print(f"Output file: {output_path}")

    scorer = MusiqScorer(musiq_source, checkpoint, num_classes)
    records: List[Dict[str, Any]] = []
    successful_scores: List[float] = []

    base_dir = input_path if input_path.is_dir() else input_path.parent
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}", flush=True)
        record: Dict[str, Any] = {
            "image": str(image_path),
            "relative_image": os.path.relpath(image_path, base_dir),
            "checkpoint": str(checkpoint),
            "num_classes": num_classes,
            "status": "ok",
            "score": None,
            "error": None,
        }
        try:
            score = scorer.score_image(image_path)
            record["score"] = score
            successful_scores.append(score)
        except Exception as exc:  # Keep a bad file from hiding other results.
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  Skipped: {record['error']}", file=sys.stderr)
        records.append(record)

    write_jsonl(output_path, records)
    csv_path = output_path.with_suffix(".csv")
    write_csv(csv_path, records)

    score_summary = aggregate_scores(successful_scores)
    summary = {
        "input": str(input_path),
        "checkpoint": str(checkpoint),
        "num_classes": num_classes,
        "musiq_source": str(musiq_source),
        "total_images": len(images),
        "successful_images": len(successful_scores),
        "failed_images": len(images) - len(successful_scores),
        "score_summary": score_summary,
        "records_jsonl": str(output_path),
        "records_csv": str(csv_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(score_summary, ensure_ascii=False, indent=2))
    print(f"Per-image results: {output_path}")
    print(f"CSV copy: {csv_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> int:
    args = parse_args()

    # This must happen before importing JAX/TensorFlow through MUSIQ.
    if args.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    musiq_source = Path(args.musiq_source).expanduser().resolve()
    images = discover_images(input_path, args.recursive)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        images = images[: args.max_images]

    requested_checkpoints: List[Tuple[str, Path, int]] = []
    if args.all_four:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        requested_checkpoints.extend(
            (name, checkpoint_dir / filename, classes)
            for name, filename, classes in FINAL_CHECKPOINTS
        )
    if args.ckpt:
        for ckpt_text in args.ckpt:
            ckpt_path = Path(ckpt_text).expanduser().resolve()
            if args.num_classes == "auto":
                classes = 10 if "ava" in ckpt_path.stem.lower() else 1
            else:
                classes = int(args.num_classes)
            requested_checkpoints.append((checkpoint_name(ckpt_path), ckpt_path, classes))
    if not requested_checkpoints:
        raise ValueError("Specify --ckpt, or use --all-four to evaluate the four official checkpoints")

    # Preserve user order but avoid evaluating the same checkpoint twice.
    unique_checkpoints: List[Tuple[str, Path, int]] = []
    seen = set()
    for name, checkpoint, classes in requested_checkpoints:
        key = str(checkpoint)
        if key not in seen:
            unique_checkpoints.append((name, checkpoint, classes))
            seen.add(key)

    summaries: Dict[str, Any] = {}
    multiple = len(unique_checkpoints) > 1
    for name, checkpoint, classes in unique_checkpoints:
        result_path = output_path_for_checkpoint(output_path, name, multiple)
        summaries[name] = evaluate_one_checkpoint(
            images=images,
            input_path=input_path,
            output_path=result_path,
            checkpoint=checkpoint,
            num_classes=classes,
            musiq_source=musiq_source,
        )

    if multiple:
        combined_path = output_path.with_suffix(".all.summary.json")
        combined = {
            "input": str(input_path),
            "musiq_source": str(musiq_source),
            "checkpoint_count": len(summaries),
            "checkpoints": summaries,
        }
        combined_path.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Combined summary for the four checkpoints: {combined_path}")

    return 0 if any(
        summary["successful_images"] > 0 for summary in summaries.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
