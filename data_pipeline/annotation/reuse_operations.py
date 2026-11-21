"""Bind accepted operation labels to an input manifest for long recaptioning."""

import argparse
import hashlib
import json
from pathlib import Path

from manifest_io import read_manifest, relocate_paths


def prepare(current, previous, output):
    frozen = read_manifest(current)
    history = {record["id"]: record for record in read_manifest(previous)}
    source_hash = hashlib.sha256(previous.read_bytes()).hexdigest()
    result = []
    for row in frozen:
        prior = history.get(row["id"])
        if prior is None or any(
            prior.get(field) != row.get(field) for field in ("domain", "video_sha", "source_frames")
        ):
            raise ValueError(f"No source-matched prior classification for {row['id']}")
        annotation = prior.get("caption_annotation", prior)
        operation = annotation.get("operation") if isinstance(annotation, dict) else None
        if not isinstance(operation, dict) or not operation.get("code"):
            raise ValueError(f"Prior classification missing for {row['id']}")
        relocated = relocate_paths(row, current.parent, output.parent)
        relocated["previous_operation"] = operation
        relocated["previous_operation_provenance"] = {
            "source_case_id": row["id"],
            "source_manifest_sha256": source_hash,
            "source_video_sha256": prior["video_sha"],
        }
        sidecar = prior.get("caption_sidecar")
        sidecar = sidecar if isinstance(sidecar, dict) else {}
        if prior.get("caption_sha256") or sidecar.get("sha256"):
            relocated["previous_operation_provenance"]["source_caption_sha256"] = (
                prior.get("caption_sha256") or sidecar["sha256"]
            )
        if annotation.get("model"):
            relocated["previous_operation_provenance"]["model"] = annotation["model"]
        result.append(relocated)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in result)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = prepare(args.input, args.previous, args.output)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({"prior_labels_bound": count}))


if __name__ == "__main__":
    main()
