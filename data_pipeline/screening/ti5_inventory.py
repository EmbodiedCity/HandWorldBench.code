"""Discover TI5 dexterous-hand candidates from raw episode metadata."""
import hashlib
import json
from pathlib import Path


def parse(path):
    raw = path.read_bytes()
    source = json.loads(raw)
    metadata = source.get("metadata") or {}
    observations = metadata.get("observations") or {}
    cameras = metadata.get("cameras") or {}
    hand = "hand" in observations and "gripper" not in observations
    task = source.get("task_name")
    row = {
        "source_metadata_path": str(path), "task": task, "dataset": source.get("dataset_name"),
        "session_id": source.get("session_id"), "episode_id": source.get("episode_id"),
        "instruction": source.get("instruction"),
        "frames": source.get("valid_sample_count"), "fps": source.get("fps"),
        "has_dexterous_hand_metadata": hand,
        "status": "other_embodiment",
    }
    if not hand:
        return row
    files = {p.name: p for p in path.parent.iterdir() if p.is_file()}
    head_name = (cameras.get("cam_head") or {}).get("file", "cam_head.mp4")
    hdf5 = [p for p in files.values() if p.suffix in {".hdf5", ".h5"}]
    video = sorted(str(p) for p in files.values() if p.suffix.lower() == ".mp4")
    if head_name not in files or len(hdf5) != 1 or not cameras:
        return {**row, "status": "source_files_missing_or_ambiguous"}
    if any(camera.get("file") not in files for camera in cameras.values() if camera.get("file")):
        return {**row, "status": "source_files_missing_or_ambiguous"}
    if not task or not row["frames"] or not row["fps"]:
        return {**row, "status": "episode_metadata_incomplete"}
    return {
        **row, "status": "metadata_candidate", "source_episode_path": str(path.parent),
        "source_video_path": str(files[head_name]), "source_hdf5_path": str(hdf5[0]),
        "source_camera_paths": video, "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "source_json_paths": sorted(str(p) for p in files.values() if p.suffix == ".json"),
        "camera_info": cameras, "semantic_review": "pending",
    }
