"""Screen MCAP episodes with the recovered geometry and RGB profile, then verify exports."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import os

from ego_pixels import fingerprint, read_pixels, select_verified
from handclip.curate import export_job, load_config, source_job
from handclip.recording_isolation import load_policy
from robot_io import atomic, sha, write_jsonl


def frozen_input(path, expected_hash):
    if path is None:
        return {}, {}
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("frozen Ego manifest checksum mismatch")
    intervals = defaultdict(list)
    rows = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        start, end = row["global_frames"]
        if start < 0 or end - start != row["frames"]:
            raise ValueError(f"invalid frozen interval for {row['id']}")
        if row["id"] in rows:
            raise ValueError(f"duplicate frozen clip {row['id']}")
        rows[row["id"]] = row
        intervals[row["group"]].append((start, end))
    return dict(intervals), rows


def scan_source(request):
    source, root, config, contract = request
    before = source.stat()
    key = hashlib.sha256(str(source).encode()).hexdigest()
    receipt_path = root / "qc" / f"{key}.json"
    source_stat = {"size": before.st_size, "mtime_ns": before.st_mtime_ns}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_stat"] == source_stat and receipt["contract"] == contract:
            if not receipt["rows"] or sha(source) == receipt["source_sha256"]:
                for row in receipt["rows"]:
                    for role, checksum in (("video", "video_sha"), ("first_frame", "first_frame_sha256"),
                                           ("annotation", "annotation_sha256")):
                        if sha(root / row[role]) != row[checksum]:
                            raise ValueError(f"cached Ego {role} changed for {row['id']}")
                return receipt
    report = source_job(str(source), config)
    if report["status"] == "error":
        raise RuntimeError(report["error"])
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed during scan {source.name}")
    rows = []
    for candidate in report["candidates"]:
        result = export_job(root, candidate, config)
        artifact = result["export"]
        validation = result["verification"]
        dest = Path(artifact["video"]).parent
        frames = validation["frame_count"]
        offset = candidate["source_frame_offset"]
        start, end = candidate["start_frame"], candidate["end_frame"]
        global_frames = [offset + start, offset + end]
        video = os.path.relpath(artifact["video"], root)
        video_sha = validation["video_sha256"]
        source_sha = artifact["source_sha256"]
        group = candidate["recording_id"]
        key = json.dumps([group, global_frames, source_sha], ensure_ascii=False, separators=(",", ":"))
        clip_id = "ego_" + hashlib.sha256(key.encode()).hexdigest()[:24]
        rows.append(dict(
            id=clip_id, domain="ego", group=group,
            source_identity=group, frames=frames, fps=candidate["fps"],
            source_frames=[start, end], global_frames=global_frames,
            source=os.path.relpath(source, root), source_sha=source_sha,
            video=video, video_sha=video_sha,
            videos={"ego": dict(path=video, sha256=video_sha, decoded_frames=frames)},
            fp=fingerprint(root / video, frames),
            annotation=os.path.relpath(artifact["annotations"], root),
            annotation_sha256=artifact["checksums"]["annotations.npz"],
            first_frame=os.path.relpath(artifact["first_frame"], root),
            first_frame_sha256=artifact["checksums"]["first_frame.png"],
            artifact=os.path.relpath(dest / "artifact.json", root),
            source_frame_hashes=os.path.relpath(dest / "source_frame_hashes.json", root),
            camera_pose_measured=True, quality_status="technical_pass", human_review="pending",
        ))
    receipt = dict(
        source=os.path.relpath(source, root), source_stat=source_stat,
        source_sha256=report.get("source_sha256"), contract=contract, rows=rows,
        stage=report["stage"], rejections=report.get("rejections", {}),
    )
    atomic(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True, help="one MCAP path per line, relative to the inventory")
    parser.add_argument("--recording-policy", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--frozen-pixels", type=Path, help="bound dHash receipts for the frozen manifest")
    parser.add_argument("--holdout-pixels", type=Path, help="dHash receipts for benchmark holdout videos")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("ego_qc.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (args.frozen is None) != (args.frozen_sha256 is None) or args.workers < 1:
        parser.error("provide both frozen inputs when screening an addition, and a positive worker count")
    if args.frozen_pixels and not args.frozen:
        parser.error("--frozen-pixels requires --frozen")
    sources = [(args.inventory.parent / line.strip()).resolve()
               for line in args.inventory.read_text().splitlines() if line.strip()]
    if not sources or len(sources) != len(set(sources)):
        parser.error("inventory must contain unique MCAP file references")
    blocked, policy_id = load_policy(args.recording_policy)
    config = load_config(args.config)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("qc", "cache", "budget", "samples"):
            (root / name).mkdir(exist_ok=True)
        config["runtime"].update(
            workers=args.workers, output_root=str(root),
            cache_root=str(root / "cache"), cache_budget_root=str(root / "budget"),
            sample_root=str(root / "samples"),
        )
        config["recording_isolation"] = str(args.recording_policy.resolve())
        intervals, frozen_rows = frozen_input(args.frozen, args.frozen_sha256)
        config["frozen_recording_intervals"] = intervals
        holdout = read_pixels(args.holdout_pixels)
        if any(row.get("status") != "ok" for row in holdout.values()):
            raise ValueError("benchmark pixel receipts contain unverified entries")
        frozen_pixels = read_pixels(args.frozen_pixels)
        contract = dict(
            inventory_sha256=sha(args.inventory), policy_sha256=sha(args.recording_policy),
            frozen_sha256=args.frozen_sha256, profile_sha256=sha(args.config),
            holdout_pixels_sha256=sha(args.holdout_pixels) if args.holdout_pixels else None,
            frozen_pixels_sha256=sha(args.frozen_pixels) if args.frozen_pixels else None,
            implementation_sha256=sha(__file__),
        )
        contract_file = root / "screen_contract.json"
        if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
            raise ValueError("screening input or policy changed; use a new run directory")
        atomic(contract_file, contract)
        requests = ((source, root, config, contract) for source in sources)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            receipts = list(pool.map(scan_source, requests))
        rows = [row for receipt in receipts for row in receipt["rows"]]
        if any(row["group"] in blocked for row in rows):
            raise ValueError("output contains a held-out recording")
        rows, duplicates = select_verified(
            rows, root, holdout, frozen_rows, frozen_pixels, args.frozen.parent if args.frozen else root,
        )
        write_jsonl(root / "pixel_exclusions.jsonl", duplicates)
        manifest = root / "manifest.verified.jsonl"
        write_jsonl(manifest, rows)
        atomic(root / "COMPLETE.json", dict(
            manifest=manifest.name, manifest_sha256=sha(manifest), source_count=len(sources),
            clips=len(rows), stages=Counter(receipt["stage"] for receipt in receipts),
            scope_policy=policy_id, frozen_fingerprints=len(frozen_pixels),
            benchmark_fingerprints=len(holdout), human_review="pending",
        ))


if __name__ == "__main__":
    main()
