"""Keep the frozen Ego batch and take the longest verified clips from batch two."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


ASSET_FIELDS = {
    "video", "annotation", "source", "candidate_proof", "selection_config",
    "artifact", "qc_receipt", "source_hash_receipt", "first_frame",
    "source_metadata", "source_frame_hashes", "aesthetic_receipt",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Manifest checksum mismatch for {path.name}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def read_policy(path):
    policy = json.loads(path.read_bytes())
    if policy.get("scope") != "whole_recording" or policy.get("user_confirmed") is not True:
        raise ValueError("The isolation policy must confirm whole-recording exclusion")
    raw = (path.parent / policy["recordings_file"]).read_bytes()
    if digest(raw) != policy["recordings_sha256"]:
        raise ValueError("Excluded-recording checksum mismatch")
    values = json.loads(raw)
    if (not isinstance(values, list) or not values or
            any(not isinstance(group, str) or not group for group in values) or
            len(values) != len(set(values))):
        raise ValueError("The list of excluded recordings must contain unique nonempty IDs")
    return set(values), policy.get("policy_id")


def selection_key(row):
    return -row["frames"], row["group"], row["global_frames"][0], row["id"]


def overlap(left, right):
    return min(left[1], right[1]) - max(left[0], right[0])


def verify_batch(rows, shortest):
    by_recording = defaultdict(list)
    for row in rows:
        n = row["frames"]
        local, global_frames = row["source_frames"], row["global_frames"]
        if (n < shortest or (n - 1) % 80 or local[1] - local[0] != n or
                global_frames[1] - global_frames[0] != n):
            raise ValueError(f"Invalid frame range or length for {row['id']}")
        if row["quality_status"] != "technical_pass" or not all(
            row.get(field) for field in ("id", "video_sha", "video", "annotation", "source", "group")
        ):
            raise ValueError(f"Missing source or technical acceptance for {row['id']}")
        by_recording[row["group"]].append(row)
    for group, items in by_recording.items():
        items.sort(key=lambda row: row["global_frames"])
        for previous, current in zip(items, items[1:]):
            a, b = previous["global_frames"], current["global_frames"]
            amount = overlap(a, b)
            shared_endpoint = amount == 1 and previous["source"] == current["source"] and a[1] - 1 == b[0]
            if amount > 0 and not shared_endpoint:
                raise ValueError(f"Conflicting intervals within batch for {group}")
    return by_recording


def select(frozen, verified, excluded, *, total=10000, frozen_count=8003):
    if len(frozen) != frozen_count or not 0 < frozen_count < total:
        raise ValueError("Frozen batch size does not match the requested split")
    need = total - frozen_count
    if len(verified) < need:
        raise ValueError(f"Need {need - len(verified)} additional verified clips")
    selected = sorted(verified, key=selection_key)[:need]
    first_groups = verify_batch(frozen, 81)
    verify_batch(selected, 161)
    combined = sorted([*frozen, *selected], key=selection_key)
    if len({row["id"] for row in combined}) != total or len({row["video_sha"] for row in combined}) != total:
        raise ValueError("The combined manifest repeats a clip ID or exact video")
    if any(row["group"] in excluded for row in combined):
        raise ValueError("The training selection contains an excluded benchmark recording")
    for row in selected:
        if any(overlap(row["global_frames"], old["global_frames"]) > 0
               for old in first_groups.get(row["group"], ())):
            raise ValueError(f"New clip {row['id']} shares frames with the frozen batch")
    return combined, selected


def rebase(value, original_dir, target_dir, field=None):
    if isinstance(value, dict):
        return {key: rebase(item, original_dir, target_dir, key) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item, original_dir, target_dir, field) for item in value]
    if isinstance(value, str) and value and "://" not in value:
        if field in ASSET_FIELDS or Path(value).is_absolute():
            return os.path.relpath(original_dir / value, target_dir)
    return value


def write_manifest(path, rows):
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(data)
    return digest(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--verified-sha256", required=True)
    parser.add_argument("--isolation-policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--frozen-count", type=int, default=8003)
    args = parser.parse_args()
    try:
        frozen = read_manifest(args.frozen, args.frozen_sha256)
        verified = read_manifest(args.verified, args.verified_sha256)
        excluded, policy_id = read_policy(args.isolation_policy)
        combined, selected = select(frozen, verified, excluded, total=args.total, frozen_count=args.frozen_count)
        args.out.mkdir(parents=True, exist_ok=False)
        selected_ids = {row["id"] for row in selected}
        first_dir = args.frozen.parent.resolve()
        new_dir = args.verified.parent.resolve()
        out_dir = args.out.resolve()
        merged_name, subset_name = "manifest.ego.jsonl", "manifest.batch02.jsonl"
        merged_sha = write_manifest(out_dir / merged_name, [
            rebase(row, new_dir if row["id"] in selected_ids else first_dir, out_dir) for row in combined
        ])
        subset_sha = write_manifest(out_dir / subset_name, [rebase(row, new_dir, out_dir) for row in selected])
        receipt = {
            "manifest": merged_name, "sha256": merged_sha, "count": len(combined),
            "batch2_subset": subset_name, "batch2_subset_sha256": subset_sha,
            "first_count": len(frozen), "second_count_used": len(selected),
            "frozen_input_sha256": args.frozen_sha256, "verified_input_sha256": args.verified_sha256,
            "isolation_policy": policy_id, "hist": dict(sorted(Counter(row["frames"] for row in combined).items())),
            "manual_review": "pending",
        }
        (out_dir / "RELEASE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
