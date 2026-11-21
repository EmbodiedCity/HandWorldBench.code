"""Read, validate and relocate file paths in screening and annotation manifests."""

import json
import os
import re
from pathlib import Path


ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
SHA_PATTERN = re.compile(r"[a-f0-9]{64}")
PATH_KEYS = {
    "video", "first_frame", "first_frame_path", "caption_path", "parquet", "hdf5",
    "metadata", "mcap", "render_metadata_tar", "source_video_path",
    "source_hdf5_path", "source_metadata_path", "source_path",
    "source_raw_path", "annotation", "artifact", "source", "source_frame_hashes",
    "source_episode_path", "source_camera_paths", "source_json_paths",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_manifest(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("empty manifest")
    ids = set()
    for row in rows:
        clip_id = row["id"]
        frames = row["frames"]
        start, end = row["source_frames"]
        if not ID_PATTERN.fullmatch(clip_id) or clip_id in {".", ".."} or clip_id in ids:
            raise ValueError(f"duplicate or invalid clip ID {clip_id!r}")
        if row["domain"] not in {"ego", "handson", "ti5"}:
            raise ValueError(f"unknown domain for {clip_id!r}")
        if (type(frames) is not int or type(start) is not int or type(end) is not int
                or start < 0 or frames < 81 or (frames - 1) % 80 or end - start != frames):
            raise ValueError(f"invalid continuous frame interval for {clip_id!r}")
        if not row["video"] or not SHA_PATTERN.fullmatch(row["video_sha"]):
            raise ValueError(f"invalid main video reference for {clip_id!r}")
        ids.add(clip_id)
    return rows


def relocate_paths(value, source_dir, output_dir):
    """Rebase known file references. Preserve opaque IDs and relative archive members."""
    source_dir = Path(source_dir).resolve()
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
    def visit(item, parent_key=None, parent=None):
        if isinstance(item, list):
            return [visit(child, parent_key, parent) for child in item]
        if not isinstance(item, dict):
            is_reference = parent_key in PATH_KEYS or (parent_key == "path" and "sha256" in parent)
            if is_reference and isinstance(item, str) and item and "://" not in item:
                path = (source_dir / item).resolve()
                return str(path) if output_dir is None else os.path.relpath(path, output_dir)
            return item
        return {key: visit(child, key, item) for key, child in item.items()}
    return visit(value)
