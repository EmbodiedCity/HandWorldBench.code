"""GPU discovery with explicit logical-to-physical device binding."""
import csv
import itertools
import os
import subprocess
from dataclasses import dataclass

@dataclass(frozen=True)
class Device:
    index: int
    physical_index: int
    uuid: str
    total_mib: float
    free_mib: float

def inventory(physical=None, visible=None):
    if physical is None:
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
        physical = [dict(index=int(r[0]), uuid=r[1].strip(), total_mib=float(r[2]), free_mib=float(r[3])) for r in csv.reader(text.splitlines())]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    if visible is None:
        selected = physical
    elif visible.strip() in ("", "-1"):
        selected = []
    else:
        selected = []
        for token in visible.split(","):
            token = token.strip()
            matches = [p for p in physical if str(p["index"]) == token or p["uuid"].startswith(token)]
            if len(matches) != 1 or matches[0] in selected:
                raise ValueError(f"Cannot resolve visible GPU: {token}")
            selected.append(matches[0])
    return [Device(i, p["index"], p["uuid"], p["total_mib"], p["free_mib"]) for i, p in enumerate(selected)]

def select_devices(spec="auto", available=None):
    available = inventory() if available is None else available
    selected = list(range(len(available))) if spec == "auto" else [int(x) for x in spec.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(available) for i in selected):
        raise ValueError("GPU selection must be a nonempty unique subset of visible devices")
    return [available[i] for i in selected]

def placements(devices, count, required_mib, busy=(), resident=(), load_seconds=0, infer_seconds=0):
    """Return feasible device groups sorted by completion cost and device IDs."""
    free = [d for d in devices if d.index not in busy and d.free_mib >= required_mib]
    choices = []
    for group in itertools.combinations(free, count):
        ids = tuple(d.index for d in group)
        choices.append((infer_seconds + (0 if ids in resident else load_seconds), ids))
    return sorted(choices)

