"""Generate a moving-pattern video and a portable 81-frame caption input."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    video = args.out / "pattern.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=128x96:rate=30", "-frames:v", "81", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    row = {
        "id": "synthetic-motion", "domain": "ego", "frames": 81, "fps": 30,
        "source_frames": [0, 81], "video": video.name, "video_sha": video_sha,
        "videos": {"ego": {"path": video.name, "sha256": video_sha, "decoded_frames": 81}},
        "synthetic": True,
    }
    (args.out / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
