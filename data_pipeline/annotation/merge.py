"""Validate a completed caption journal and join annotations to the screening manifest."""

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

from caption_contract import check_caption
from manifest_io import read_jsonl, read_manifest, relocate_paths


def merge(input_path, run_dir, output_path):
    if not __debug__:
        raise RuntimeError("caption validation requires Python assertions to be enabled")
    rows = read_manifest(input_path)
    contract = json.loads((run_dir / "run_contract.json").read_text())
    expected = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if contract["queue_sha256"] != expected:
        raise ValueError("screening manifest differs from the input bound to this caption run")
    results = {}
    for item in read_jsonl(run_dir / "results.jsonl"):
        clip_id = item["id"]
        if clip_id in results or item["status"] != "annotated_pending_review":
            raise ValueError(f"duplicate or incomplete caption result for {clip_id!r}")
        results[clip_id] = item
    if set(results) != {row["id"] for row in rows}:
        raise ValueError("screening and caption IDs differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=output_path.parent, encoding="utf-8", delete=False,
    ) as output:
        temporary_path = Path(output.name)
        try:
            for row in rows:
                entry = results[row["id"]]
                caption_path = (run_dir / entry["caption"]).resolve()
                if not caption_path.is_relative_to(run_dir.resolve()):
                    raise ValueError("caption journal references a path outside its run")
                payload = caption_path.read_bytes()
                caption = json.loads(payload)
                check_caption(caption, row)
                if (caption["signature"] != entry["signature"]
                        or caption["contract"] != contract["annotation_contract"]
                        or caption["human_review"] != "pending"):
                    raise ValueError(f"mismatched caption provenance for {row['id']!r}")
                merged = relocate_paths(row, input_path.parent, output_path.parent)
                merged.update(
                    caption_path=os.path.relpath(caption_path, output_path.parent),
                    caption_sha256=hashlib.sha256(payload).hexdigest(),
                    task_text=caption["task_text"],
                    segments=caption["segments"],
                    operation=caption["operation"],
                    operation_provenance=caption["operation_provenance"],
                    review_flags=caption["review_flags"],
                    annotation_status="annotated_pending_review",
                    human_review="pending",
                )
                output.write(json.dumps(merged, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            lock_path = output_path.with_name(output_path.name + ".lock")
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if output_path.exists():
                    raise FileExistsError(output_path)
                os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"annotated": merge(args.input, args.run_dir, args.output)}))


if __name__ == "__main__":
    main()
