"""Configured storage and model resources for inference backends."""
import os
from pathlib import Path
from handworldbench.runtime.storage import read,write
from handworldbench.observations.video import stage,decode
CONFIG=read(os.environ['HANDWORLDBENCH_RUNTIME_CONFIG'])['runtime']
ENGINE=Path(__file__).resolve().parents[1]/'_vendor/engine'
S=Path(CONFIG['cache_root'])/'native'
S.mkdir(parents=True,exist_ok=True)
def stage_video(row):return stage(row['video'],row['video_sha256'],CONFIG['cache_root'])[0]
def decoded_frames(row):return decode(row['video'],row['video_sha256'],CONFIG['cache_root'],row.get('role','prediction'),count=CONFIG.get('sampling',{}).get('frames',16))
