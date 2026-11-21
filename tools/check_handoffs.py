"""Check identity and provenance at the training and benchmark handoffs."""

import argparse
import json
from pathlib import Path


def read_rows(path, container):
    content = path.read_text(encoding="utf-8")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get(container, [value])
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path.name} must contain a nonempty list of records")
    return value


def index_rows(rows, label, identifiers, *, with_models=False):
    index = {}
    for row in rows:
        case = next((row[key] for key in identifiers if row.get(key)), None)
        if not isinstance(case, str) or not case.strip():
            raise ValueError(f"{label} has a record without a case ID")
        key = (row.get("model"), case) if with_models else case
        if key in index:
            raise ValueError(f"{label} repeats record {key}")
        index[key] = row
    return index


def require_subset(items, population, label):
    missing = sorted(set(items) - set(population))
    if missing:
        raise ValueError(f"{label} contains {len(missing)} unknown cases, including {missing[:5]}")


def check_handoffs(captions=None, controls=None, predictions=None, catalog=None):
    if (captions is None) != (controls is None):
        raise ValueError("Supply both the caption and control manifests for a training check")
    if (predictions is None) != (catalog is None):
        raise ValueError("Supply both predictions and the evaluation catalog for a benchmark check")
    result = {}
    if captions is not None:
        caption_index = index_rows(read_rows(captions, "records"), "Captions", ("id", "case_id"))
        control_index = index_rows(read_rows(controls, "records"), "Controls", ("clip_id", "case_id"))
        require_subset(control_index, caption_index, "Controls")
        compared = 0
        for case, row in control_index.items():
            original_sha = row.get("original_video_sha256")
            source_sha = caption_index[case].get("video_sha") or caption_index[case].get("video_sha256")
            if original_sha and source_sha:
                if original_sha != source_sha:
                    raise ValueError(f"Control source video SHA-256 differs from its caption source for {case}")
                compared += 1
        result["training"] = {
            "caption_cases": len(caption_index),
            "control_cases": len(control_index),
            "source_hashes_compared": compared,
        }
    if predictions is not None:
        catalog_index = index_rows(read_rows(catalog, "cases"), "Evaluation catalog", ("case_id",))
        prediction_index = index_rows(
            read_rows(predictions, "predictions"), "Predictions", ("case_id",), with_models=True
        )
        require_subset((case for _, case in prediction_index), catalog_index, "Predictions")
        for row in prediction_index.values():
            if not isinstance(row.get("video"), str) or not row["video"]:
                raise ValueError(f"Prediction for {row['case_id']} has no video path")
        result["benchmark"] = {
            "catalog_cases": len(catalog_index),
            "prediction_records": len(prediction_index),
            "prediction_cases": len({case for _, case in prediction_index}),
        }
    if not result:
        raise ValueError("Supply at least one pair of manifests")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    try:
        result = check_handoffs(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
