"""Shared length allocation and durable native-video export."""
import hashlib, json, os, shutil, subprocess, tempfile, uuid
from fractions import Fraction
from pathlib import Path

FRAME_MAPPING = "decoded_source_prefix_in_order_v1"
ROBOT_VIEWS = {"handson": ("front", "wrist_left", "wrist_right"), "ti5": ("head",)}

FILE_FIELDS = {
 "path", "video", "first_frame", "parquet", "metadata", "hdf5",
 "render_metadata_tar", "source_episode_path", "source_video_path",
 "source_hdf5_path", "source_metadata_path", "source_camera_paths", "source_json_paths",
}

def _convert_refs(value, transform, field=None, parent=None):
 if isinstance(value,dict):
  return {k:_convert_refs(v,transform,k,value) for k,v in value.items()}
 if isinstance(value,list):
  return [_convert_refs(v,transform,field,parent) for v in value]
 if isinstance(value,str) and value and field in FILE_FIELDS:
  if field!='path' or 'sha256' in parent:
   return transform(value)
 return value

def relative_record(row,root):
 return _convert_refs(row,lambda path:os.path.relpath(path,root))

def absolute_record(row,root):
 return _convert_refs(row,lambda path:str((root/path).resolve()))

def source_key(row):
 return row.get('episode_uid') or row.get('source_episode_key') or row['source_metadata_path']

def case_id(domain,row):
 return domain+'_native_'+hashlib.sha256(source_key(row).encode()).hexdigest()[:24]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 b=json.dumps(obj,ensure_ascii=False,indent=2).encode()
 with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as f:f.write(b);f.flush();os.fsync(f.fileno());tmp=f.name
 os.replace(tmp,path)
 if path.read_bytes()!=b:raise IOError('publish readback mismatch')
def write_jsonl(path,rows):
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno());tmp=f.name
 expected=sha(tmp);os.replace(tmp,path)
 if sha(path)!=expected:raise IOError('JSONL hash mismatch')
def allocate(hist,n):
 if not hist:return {}
 total=sum(hist.values());out={k:n*v//total for k,v in hist.items()}
 for k in sorted(hist,key=lambda k:(-(n*hist[k]%total),k))[:n-sum(out.values())]:out[k]+=1
 return out
def bucket(r):return ((int(r['frames'])-1)//80)*80+1


def frame_rate(fps):
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    if rate <= 0:
        raise ValueError("Frame rate must be positive")
    return rate


def encode(src, dest, n, fps, gamma=1.):
    rate = frame_rate(fps)
    if n < 1:
        raise ValueError("Export must include at least one frame")
    time_base = f"{rate.denominator}/{rate.numerator}"
    # Put decoded source frame i at i / fps before declaring the output rate.
    # VFR gaps in the source must not cause the output to duplicate or drop frames.
    filters = (
        f"trim=start_frame=0:end_frame={n},settb=expr={time_base},"
        f"setpts=N,fps=fps={rate}:start_time=0:round=near"
    )
    if gamma != 1.:
        filters += f",eq=gamma={gamma}"

    with tempfile.TemporaryDirectory(prefix="robot_encode_", dir=Path(src).parent) as directory:
        encoded = Path(directory) / "clip.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-xerror", "-y", "-threads", "1",
            "-filter_threads", "1", "-i", str(src), "-map", "0:v:0", "-an", "-vf", filters,
            "-frames:v", str(n), "-fps_mode", "passthrough",
            "-enc_time_base", time_base.replace("/", ":"), "-c:v", "libx264",
            "-crf", "18", "-preset", "fast", "-threads", "1", "-pix_fmt", "yuv420p", str(encoded),
        ], check=True, timeout=1800)
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate,start_pts",
            "-of", "json", str(encoded),
        ], timeout=1800)
        video = json.loads(probe)["streams"][0]
        if int(video["nb_read_frames"]) != n:
            raise ValueError("export decoded count mismatch")
        if Fraction(video["avg_frame_rate"]) != rate or video.get("start_pts") != 0:
            raise ValueError("export playback timing mismatch")
        expected = sha(encoded)
        upload = dest.with_name(dest.name + ".upload." + uuid.uuid4().hex)
        try:
            with encoded.open("rb") as incoming, upload.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 4 << 20)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if sha(upload) != expected:
                raise IOError("encoded video publish mismatch")
            os.replace(upload, dest)
        finally:
            upload.unlink(missing_ok=True)

    return dict(path=str(dest), sha256=expected, decoded_frames=n, width=video["width"],
                height=video["height"], frame_mapping=FRAME_MAPPING, frame_rate=str(rate))


def verify_clip_receipt(record, source, domain):
    frames = bucket(source)
    rate = str(frame_rate(source["fps"]))
    if (record["domain"] != domain or record["id"] != case_id(domain, source)
            or record["source_identity"] != source_key(source)
            or record["source_frames"] != [0, frames] or record["frames"] != frames
            or record["source_raw_frames"] != source["frames"]
            or str(frame_rate(record["fps"])) != rate):
        raise ValueError("cached clip no longer matches the source interval")
    views = record["videos"]
    if set(views) != set(ROBOT_VIEWS[domain]):
        raise ValueError("cached clip has an unexpected view set")
    primary = views[ROBOT_VIEWS[domain][0]]
    if (record["video"], record["video_sha"]) != (primary["path"], primary["sha256"]):
        raise ValueError("cached clip primary view disagrees with its video receipt")
    for view in views.values():
        if (view.get("frame_mapping") != FRAME_MAPPING or view.get("frame_rate") != rate
                or view["decoded_frames"] != frames):
            raise ValueError("cached video needs re-encoding under the current frame-mapping policy")
        if sha(view["path"]) != view["sha256"]:
            raise ValueError("cached video SHA-256 changed")
    if sha(record["first_frame"]) != record["first_frame_sha256"]:
        raise ValueError("cached first frame SHA-256 changed")


def verify_ti5_source_receipt(record, source):
    native = record["native_ref"]
    if (source.get("status") != "metadata_candidate"
            or native["metadata"] != source["source_metadata_path"]
            or native["hdf5"] != source["source_hdf5_path"]
            or native["metadata_sha256"] != source["metadata_sha256"]
            or sha(native["metadata"]) != native["metadata_sha256"]):
        raise ValueError("cached TI5 metadata or source paths changed")
    if (record["raw_video_sha256"] != sha(source["source_video_path"])
            or sha(native["hdf5"]) != native.get("hdf5_sha256")):
        raise ValueError("cached TI5 video or HDF5 source SHA-256 changed")
