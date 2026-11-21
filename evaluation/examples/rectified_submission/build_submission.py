"""Build a HandWorldBench prediction manifest for a video with explicit mappings."""

import argparse
import json
import os
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_mapping(path, kinds):
    mapping = read_json(path)
    if not isinstance(mapping, dict) or mapping.get("kind") not in kinds:
        raise ValueError(f"{path}: expected a mapping with kind in {', '.join(kinds)}")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--model", required=True, help="Name for your model's evaluation run")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--spatial-mapping", required=True, type=Path)
    parser.add_argument("--time-mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        dataset = args.dataset.resolve(strict=True)
        dataset_config = read_json(dataset / "dataset.json")
        catalog = read_json(dataset / dataset_config["catalog"])
        case = next((row for row in catalog if row["case_id"] == args.case_id), None)
        if case is None:
            raise ValueError(f"Unknown case ID: {args.case_id}")
        model = args.model.strip()
        if not model:
            raise ValueError("The evaluation run name must be nonempty")
        video = args.video.resolve(strict=True)
        if not video.is_file():
            raise ValueError(f"Expected a generated video file: {video}")

        spatial = read_mapping(
            args.spatial_mapping, ("identity", "affine", "opencv", "opencv_calibrated", "fisheye")
        )
        timeline = read_mapping(args.time_mapping, ("identity", "affine", "points"))
        if spatial["kind"] in {"opencv_calibrated", "fisheye"}:
            if not case.get("geometry", {}).get("camera_points_key"):
                raise ValueError(f"{args.case_id} has no reference camera-space 3D points")

        record = {
            "case_id": args.case_id,
            "model": model,
            "video": Path(os.path.relpath(video, dataset)).as_posix(),
            "spatial_mapping": spatial,
            "time_mapping": timeline,
        }
        payload = json.dumps([record], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
