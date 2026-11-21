"""Audit Ego clips with the frozen three-frame dHash and longest-interval policy."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def compatible(left, right):
    a, b = left["global_frames"], right["global_frames"]
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    return overlap <= 0 or (overlap == 1 and left["source"] == right["source"]
                            and (a[1] - 1 == b[0] or b[1] - 1 == a[0]))


def selection_key(row):
    return -row["frames"], row["group"], row["global_frames"][0], row["id"]


def pack(rows):
    """Use lexicographic longest-bin counts within each recording."""
    rows = list(rows)
    base = len(rows) + 1
    weights = {length: base ** i for i, length in enumerate(sorted({r["frames"] for r in rows}))}
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    selected = []
    for items in groups.values():
        items.sort(key=lambda row: (row["global_frames"][1], row["global_frames"][0], row["id"]))
        scores, previous = [], []
        for index, row in enumerate(items):
            allowed = (j for j in range(index) if compatible(items[j], row))
            best = max(allowed, key=scores.__getitem__, default=-1)
            scores.append(weights[row["frames"]] + (scores[best] if best >= 0 else 0))
            previous.append(best)
        index = max(range(len(items)), key=scores.__getitem__)
        while index >= 0:
            selected.append(items[index])
            index = previous[index]
    return sorted(selected, key=selection_key)


def fingerprint(path, frames):
    filt = f"select='eq(n,0)+eq(n,{frames // 2})+eq(n,{frames - 1})',scale=9:8:flags=area,format=gray"
    pixels = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
        "-vf", filt, "-threads", "1", "-fps_mode", "passthrough",
        "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
    ], check=True, capture_output=True).stdout
    if len(pixels) != 216:
        raise ValueError("missing one or more fingerprint frames")
    return [sum((pixels[i + y * 9 + x] > pixels[i + y * 9 + x + 1]) << (8 * y + x)
                for y in range(8) for x in range(8)) for i in (0, 72, 144)]


def read_pixels(path):
    if not path:
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (not rows or len({row["id"] for row in rows}) != len(rows)
            or any(len(row["fp"]) != 3 or any(type(bits) is not int or not 0 <= bits < 2**64
                                                for bits in row["fp"]) for row in rows)):
        raise ValueError(f"invalid fingerprint receipt {path.name}")
    return {row["id"]: row for row in rows}


def sequence_digest(row, root):
    video = root / row["video"]
    artifact = json.loads((video.parent / "artifact.json").read_text())
    checksums = artifact["checksums"]
    data = (video.parent / "source_frame_hashes.json").read_bytes()
    if (checksums["video.mp4"] != row["video_sha"] or artifact["source_sha256"] != row["source_sha"]
            or not artifact["export_validation"]["every_decoded_rgb_frame_identical_to_source"]
            or checksums["source_frame_hashes.json"] != hashlib.sha256(data).hexdigest()):
        raise ValueError("decoded frame proof does not bind the reference clip")
    hashes = json.loads(data)
    if len(hashes) != row["frames"] or any(len(value) != 64 for value in hashes):
        raise ValueError("invalid decoded frame hashes")
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def select_verified(rows, root, holdout, frozen_rows, frozen_pixels, frozen_root):
    if frozen_pixels and (set(frozen_rows) != set(frozen_pixels) or any(
            frozen_pixels[key]["video_sha"] != row["video_sha"]
            or frozen_pixels[key]["source_sha"] != row["source_sha"]
            for key, row in frozen_rows.items())):
        raise ValueError("frozen fingerprints do not match their source manifest")
    active = {row["id"]: row for row in rows}
    excluded = []
    frozen_hashes = {row["video_sha"] for row in frozen_rows.values()}
    while True:
        selected = pack(active.values())
        reference = {}
        index = defaultdict(set)
        hashes = set(frozen_hashes)

        def add(key, bits, row=None, base=None):
            reference[key] = bits, row, base
            for part in range(5):
                index[part, (bits[0] >> (13 * part)) & 8191].add(key)

        for key, evidence in holdout.items():
            add(key, evidence["fp"])
        for key, evidence in frozen_pixels.items():
            add(key, evidence["fp"], frozen_rows[key], frozen_root)
        rejected = []
        for row in selected:
            key, bits = row["id"], row["fp"]
            near = set().union(*(index[part, (bits[0] >> (13 * part)) & 8191] for part in range(5)))
            blocking = []
            for match in sorted(near):
                previous, other, base = reference[match]
                if any((a ^ b).bit_count() > 4 for a, b in zip(bits, previous)):
                    continue
                if (other is None or row["group"] != other["group"] or not compatible(row, other)
                        or sequence_digest(row, root) == sequence_digest(other, base)):
                    blocking.append(match)
            if key in frozen_rows or row["video_sha"] in hashes or blocking:
                rejected.append(key)
                excluded.append(dict(id=key, reason="duplicate_clip_or_pixels", matches=blocking))
            else:
                hashes.add(row["video_sha"])
                add(key, bits, row, root)
        if not rejected:
            return selected, excluded
        for key in rejected:
            active.pop(key)
