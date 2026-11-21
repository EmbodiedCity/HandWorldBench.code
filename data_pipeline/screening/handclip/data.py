"""Read egoscale MCAP without retaining decoded video or changing source files.

The camera log timestamps define the frame axis. Hand frame indices are local to
the episode; available payload timestamps must agree within one millisecond.
Missing samples remain invalid rather than shortening or interpolating the axis.
"""
from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np

VIDEO_TOPIC = "/top-left-camera/image-raw"
CAMERA_TOPIC = "/top-left-camera/camera-info"
POSE_TOPIC = "/ego/vio/pose"
HAND_TOPICS = ("/vision/hand_left/skeleton", "/vision/hand_right/skeleton")
TIME_TOLERANCE_NS = 1_000_000
JOINT_NAMES = ["wrist"] + [
    f"{finger}_{suffix}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for suffix in ("1", "2", "3", "tip")
]
_ANNEX_B_START = re.compile(b"\x00\x00(?:\x00)?\x01")


class DataError(ValueError):
    """The source cannot be safely interpreted on a single camera frame axis."""


class _Bits:
    def __init__(self, nal_payload):
        self.data = nal_payload.replace(b"\x00\x00\x03", b"\x00\x00")
        self.position = 0

    def read(self, count=1):
        if count < 0 or self.position + count > len(self.data) * 8:
            raise DataError("Truncated H264 parameter/slice header")
        value = 0
        for _ in range(count):
            byte, bit = divmod(self.position, 8)
            value = (value << 1) | ((self.data[byte] >> (7 - bit)) & 1)
            self.position += 1
        return value

    def ue(self):
        zeros = 0
        while self.read() == 0:
            zeros += 1
            if zeros > 31:
                raise DataError("Invalid H264 Exp-Golomb header")
        return (1 << zeros) - 1 + self.read(zeros)

    def se(self):
        value = self.ue()
        return (value + 1) // 2 if value % 2 else -(value // 2)


def _parse_sps(payload):
    bits = _Bits(payload)
    profile, _, _ = bits.read(8), bits.read(8), bits.read(8)
    sps_id = bits.ue()
    chroma = 1
    if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma = bits.ue()
        if chroma > 3:
            raise DataError("Invalid H264 chroma format")
        if chroma == 3 and bits.read():
            raise DataError("Separate H264 color planes need explicit alignment support")
        bits.ue()  # bit_depth_luma_minus8
        bits.ue()  # bit_depth_chroma_minus8
        bits.read()  # qpprime_y_zero_transform_bypass_flag
        if bits.read():
            for index in range(8 if chroma != 3 else 12):
                if bits.read():
                    last_scale = next_scale = 8
                    for _ in range(16 if index < 6 else 64):
                        if next_scale:
                            next_scale = (last_scale + bits.se() + 256) % 256
                        if next_scale:
                            last_scale = next_scale
    elif profile not in (66, 77, 88):
        raise DataError(f"Unsupported H264 SPS profile: {profile}")
    frame_num_bits = bits.ue() + 4
    if not 4 <= frame_num_bits <= 16:
        raise DataError("Invalid H264 frame_num bit width")
    poc_type = bits.ue()
    if poc_type == 0:
        if bits.ue() > 12:
            raise DataError("Invalid H264 POC bit width")
    elif poc_type == 1:
        bits.read()
        bits.se()
        bits.se()
        cycle = bits.ue()
        if cycle > 255:
            raise DataError("Invalid H264 POC cycle")
        for _ in range(cycle):
            bits.se()
    elif poc_type != 2:
        raise DataError("Invalid H264 POC type")
    bits.ue()  # max_num_ref_frames
    bits.read()  # gaps_in_frame_num_value_allowed_flag
    mb_width, map_height = bits.ue() + 1, bits.ue() + 1
    if not bits.read():
        raise DataError("Interlaced/field-coded H264 requires explicit frame alignment")
    bits.read()  # direct_8x8_inference_flag
    crop = [bits.ue() for _ in range(4)] if bits.read() else [0, 0, 0, 0]
    crop_x, crop_y = ((1, 1), (2, 2), (2, 1), (1, 1))[chroma]
    width = mb_width * 16 - (crop[0] + crop[1]) * crop_x
    height = map_height * 16 - (crop[2] + crop[3]) * crop_y
    if width < 1 or height < 1:
        raise DataError("Invalid H264 coded dimensions")
    return sps_id, dict(width=width, height=height, frame_num_bits=frame_num_bits)


class _H264PacketGuard:
    """Fail-closed source profile proven for this dataset: one progressive IDR per packet.

    Counting decoded frames cannot align B-frame decode order to skeleton indices.
    Even P-only streams need POC handling, so all predictive/multislice profiles
    remain unsupported until a separate timestamp-aware alignment implementation.
    Parsing headers is cheap and does not decode pixels.
    """
    def __init__(self):
        self.sps = {}
        self.pps = {}
        self.dimensions = None
        self.count = 0

    def check(self, packet):
        try:
            self._check(packet)
        except DataError as exc:
            raise DataError(f"unsupported codec profile: {exc}") from exc

    def _check(self, packet):
        markers = list(_ANNEX_B_START.finditer(packet))
        if not markers or any(packet[:markers[0].start()]):
            raise DataError("Camera H264 is not a validated Annex B packet")
        slices = []
        for index, marker in enumerate(markers):
            stop = markers[index + 1].start() if index + 1 < len(markers) else len(packet)
            nal = packet[marker.end():stop]
            if not nal or nal[0] & 0x80:
                raise DataError("Invalid H264 NAL header")
            kind = nal[0] & 31
            if kind == 7:
                sps_id, info = _parse_sps(nal[1:])
                self.sps[sps_id] = info
            elif kind == 8:
                bits = _Bits(nal[1:])
                pps_id, sps_id = bits.ue(), bits.ue()
                bits.read()  # entropy_coding_mode_flag
                bits.read()  # bottom_field_pic_order_in_frame_present_flag
                if bits.ue() != 0:
                    raise DataError("H264 slice groups require explicit picture alignment")
                self.pps[pps_id] = sps_id
            elif kind in (1, 5):
                bits = _Bits(nal[1:])
                first_mb, slice_type, pps_id = bits.ue(), bits.ue(), bits.ue()
                if slice_type > 9:
                    raise DataError("Invalid H264 slice type")
                if slice_type % 5 == 1:
                    raise DataError("B-frame H264 requires timestamp-aware presentation alignment")
                if kind != 5 or slice_type % 5 != 2:
                    raise DataError("Predictive/non-IDR H264 requires explicit POC alignment")
                if pps_id not in self.pps or self.pps[pps_id] not in self.sps:
                    raise DataError("H264 slice refers to an unavailable PPS/SPS")
                info = self.sps[self.pps[pps_id]]
                if bits.read(info["frame_num_bits"]) != 0 or not (nal[0] & 0x60):
                    raise DataError("Invalid IDR H264 picture header")
                bits.ue()  # idr_pic_id; require enough header bits to carry it
                dimensions = (info["width"], info["height"])
                if self.dimensions is not None and dimensions != self.dimensions:
                    raise DataError("H264 dimensions changed within episode")
                self.dimensions = dimensions
                slices.append(first_mb)
            elif kind not in (6, 9, 10, 11, 12):
                raise DataError(f"Unsupported H264 NAL type {kind} for picture alignment")
        if len(slices) != 1 or slices[0] != 0:
            raise DataError("Require exactly one complete single-slice IDR picture per H264 packet; multislice/fragmented/multiple pictures need explicit alignment")
        self.count += 1

    def summary(self):
        return dict(profile="annex_b_single_slice_progressive_idr_per_packet",
                    validated_single_idr_picture_per_packet=True,
                    packets_verified=self.count, dimensions=list(self.dimensions or ()),
                    b_frames_allowed=False, predictive_frames_allowed=False,
                    presentation_order="each packet is an independent progressive IDR picture")


def _message_classes(reader):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    summary = reader.get_summary()
    if summary is None or not summary.schemas:
        raise DataError("MCAP summary/schema missing")
    files = {}
    for schema in summary.schemas.values():
        if schema.encoding != "protobuf":
            raise DataError(f"Unsupported schema encoding: {schema.encoding}")
        descriptors = descriptor_pb2.FileDescriptorSet()
        descriptors.ParseFromString(schema.data)
        for descriptor in descriptors.file:
            serialized = descriptor.SerializeToString(deterministic=True)
            if descriptor.name in files and files[descriptor.name] != serialized:
                raise DataError(f"Conflicting protobuf file: {descriptor.name}")
            files[descriptor.name] = serialized
    pool = descriptor_pool.DescriptorPool()
    pending = dict(files)
    while pending:
        progressed = False
        for name, serialized in list(pending.items()):
            try:
                pool.AddSerializedFile(serialized)
            except (TypeError, ValueError):
                continue
            del pending[name]
            progressed = True
        if not progressed:
            raise DataError(f"Unresolved protobuf dependencies: {sorted(pending)}")
    return {
        sid: message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))
        for sid, schema in summary.schemas.items()
    }


def _messages(path, topics):
    from mcap.reader import make_reader

    with open(path, "rb") as stream:
        reader = make_reader(stream)
        classes = _message_classes(reader)
        for schema, channel, record in reader.iter_messages(topics=list(topics)):
            if schema is None or schema.id not in classes:
                raise DataError(f"Missing message schema for {channel.topic}")
            message = classes[schema.id]()
            message.ParseFromString(record.data)
            yield channel.topic, int(record.log_time), message


def _timestamp_ns(message):
    """Return None for absent payload timestamps, never invent a zero timestamp."""
    if hasattr(message, "timestamp_ns"):
        value = int(message.timestamp_ns)
        return value if value > 0 else None
    if hasattr(message, "timestamp"):
        try:
            if not message.HasField("timestamp"):
                return None
        except (AttributeError, ValueError):
            pass
        stamp = message.timestamp
        nanos = int(stamp.nanos)
        if not 0 <= nanos < 1_000_000_000:
            raise DataError("Invalid timestamp nanoseconds")
        value = int(stamp.seconds) * 1_000_000_000 + nanos
        return value if value > 0 else None
    return None


def _check_timestamp(message, log_ns, label, metadata):
    timestamp = _timestamp_ns(message)
    if timestamp is None:
        metadata["missing_payload_timestamps"][label] += 1
    elif abs(timestamp - log_ns) > TIME_TOLERANCE_NS:
        raise DataError(f"{label} payload/log timestamp disagreement")


def _frame_index(times, timestamp):
    index = int(np.searchsorted(times, timestamp))
    candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
    if not candidates:
        raise DataError("Sample outside camera timeline")
    index = min(candidates, key=lambda i: abs(int(times[i]) - timestamp))
    if abs(int(times[index]) - timestamp) > TIME_TOLERANCE_NS:
        raise DataError("Sample does not align to a camera frame within 1 ms")
    return index


def _read_metadata(path):
    metadata_path = Path(str(path) + ".metadata.json")
    if not metadata_path.is_file():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise DataError(f"Unreadable source metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise DataError("Source metadata must be an object")
    return value


def _source_identity(path, n, times, fps, annotation):
    episode_id = Path(path).parent.name
    match = re.fullmatch(
        r"(session_[A-Za-z0-9]+_\d+)_task_(\d+)_(\d+)_(\d+)_mcap_v\d+",
        episode_id,
    )
    identity = {
        "episode_id": episode_id, "recording_id": episode_id,
        "recording_confident": False, "source_frame_offset": None,
        "recording_time_origin_ns": None,
        "source_identity_reason": "unrecognized_episode_name",
    }
    if not match:
        return identity
    recording, _, start, end = match.groups()
    start, end = int(start), int(end)
    if end - start == n:
        convention = "exclusive"
    elif end - start + 1 == n:
        convention = "inclusive"
    else:
        identity["source_identity_reason"] = "directory_frame_range_mismatch"
        return identity
    if not annotation or "start_frame" not in annotation or "end_frame_exclusive" not in annotation:
        identity["source_identity_reason"] = "task_annotation_range_missing"
        return identity
    annotation_start = int(annotation["start_frame"])
    annotation_end = int(annotation["end_frame_exclusive"])
    if (annotation_start, annotation_end) not in ((0, n), (start, start + n)):
        identity["source_identity_reason"] = "task_annotation_range_mismatch"
        return identity
    identity.update(
        recording_id=recording, recording_confident=True,
        source_frame_offset=start,
        recording_time_origin_ns=int(times[0]) - round(start * 1e9 / fps),
        source_identity_reason="directory_and_annotation_verified",
        directory_end_convention=convention,
    )
    return identity


def read_episode(path, include_video=False):
    """Return aligned skeleton, VIO and source metadata; RGB is always streaming.

    ``include_video=True`` additionally retains compressed H264 packets under
    ``video_packets``. It is unnecessary for :func:`iter_video_frames`.
    """
    path = str(path)
    metadata = {
        "timestamp_tolerance_ns": TIME_TOLERANCE_NS,
        "alignment": "camera log timestamp; hand local frame index cross-check",
        "missing_payload_timestamps": collections.Counter(),
        "hand_confidence_available": False,
        "source_metadata": _read_metadata(path),
    }
    cameras, video_times, videos, poses, hand_rows, subtask_rows, annotations = [], [], [], [], [], [], []
    video_frame_id = None
    packet_guard = _H264PacketGuard()
    topics = (VIDEO_TOPIC, CAMERA_TOPIC, POSE_TOPIC, *HAND_TOPICS,
              "/task_annotation", "/subtask-annotation")
    for topic, log_ns, message in _messages(path, topics):
        if topic == CAMERA_TOPIC:
            cameras.append(message)
        elif topic == VIDEO_TOPIC:
            _check_timestamp(message, log_ns, topic, metadata)
            if str(message.format).lower() not in ("h264", "h.264") or not message.data:
                raise DataError("Expected nonempty H264 camera packets")
            packet_guard.check(message.data)
            frame_id = str(message.frame_id)
            if not frame_id or (video_frame_id is not None and frame_id != video_frame_id):
                raise DataError("Missing or changing video coordinate frame")
            video_frame_id = frame_id
            video_times.append(log_ns)
            if include_video:
                videos.append(bytes(message.data))
        elif topic in HAND_TOPICS:
            _check_timestamp(message, log_ns, topic, metadata)
            hand_rows.append((HAND_TOPICS.index(topic), log_ns, message))
        elif topic == POSE_TOPIC:
            poses.append((log_ns, message))
        elif topic == "/subtask-annotation":
            _check_timestamp(message, log_ns, topic, metadata)
            subtask_rows.append((log_ns, str(message.data)))
        elif topic == "/task_annotation":
            try:
                parsed = json.loads(message.data)
            except (TypeError, ValueError) as exc:
                raise DataError("Task annotation is not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise DataError("Task annotation must be an object")
            annotations.append(parsed)
    if len(video_times) < 2 or not cameras:
        raise DataError("Missing camera calibration or fewer than two video frames")
    times = np.asarray(video_times, dtype=np.int64)
    differences = np.diff(times)
    if np.any(differences <= 0):
        raise DataError("Duplicate or nonmonotonic camera timestamps")
    period_ns = float(np.median(differences))
    # A gap or variable frame rate would otherwise be silently time-warped on export.
    if np.max(np.abs(differences - period_ns)) > TIME_TOLERANCE_NS:
        raise DataError("Camera timeline contains gaps or variable frame intervals")
    measured_fps = 1e9 * (len(times) - 1) / int(times[-1] - times[0])
    fps = float(Fraction(measured_fps).limit_denominator(1001))
    if not 1 <= fps <= 240:
        raise DataError("Implausible camera frame rate")
    camera = cameras[0]
    camera_id = str(camera.frame_id)
    if not camera_id or camera_id != video_frame_id:
        raise DataError("Camera calibration/video coordinate frame disagreement")
    if str(camera.distortion_model).lower() != "equidistant":
        raise DataError("Only validated equidistant camera projection is supported")
    if len(camera.K) != 9 or len(camera.D) != 4:
        raise DataError("Expected 3x3 K and four equidistant coefficients")
    K = np.asarray(camera.K, dtype=float).reshape(3, 3)
    D = np.asarray(camera.D, dtype=float)
    width, height = int(camera.width), int(camera.height)
    if packet_guard.dimensions != (width, height):
        raise DataError("unsupported codec profile: H264 coded dimensions disagree with camera calibration")
    if width <= 0 or height <= 0 or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise DataError("Invalid camera calibration")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise DataError("Invalid camera intrinsic matrix")
    if abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
        raise DataError("Nonzero intrinsic skew is unsupported")
    for other in cameras[1:]:
        if (str(other.frame_id) != camera_id or str(other.distortion_model) != str(camera.distortion_model)
                or int(other.width) != width or int(other.height) != height
                or list(other.K) != list(camera.K) or list(other.D) != list(camera.D)):
            raise DataError("Camera calibration changes inside episode")
    n = len(times)
    positions = np.full((n, 2, 21, 3), np.nan)
    hand_valid = np.zeros((n, 2), dtype=bool)
    seen_hands = np.zeros((n, 2), dtype=bool)
    sources = set()
    for side, log_ns, message in hand_rows:
        index = _frame_index(times, log_ns)
        if seen_hands[index, side]:
            raise DataError("Duplicate hand sample for camera frame")
        seen_hands[index, side] = True
        if int(message.frame_index) != index:
            raise DataError("Hand frame_index disagrees with camera timestamp")
        if str(message.frame_id) != camera_id:
            raise DataError("Hand skeleton is not in calibrated camera coordinates")
        if str(message.handedness).lower() != ("left", "right")[side]:
            raise DataError("Handedness disagrees with skeleton topic")
        sources.add(str(getattr(message, "source", "unknown")))
        if not message.valid:
            continue
        points = np.asarray(message.positions_m, dtype=float)
        if points.size != 63 or not np.isfinite(points).all():
            continue
        if list(message.joint_names) != JOINT_NAMES:
            continue
        expected_tips = [4, 8, 12, 16, 20]
        if list(message.fingertip_indices) != expected_tips:
            continue
        positions[index, side] = points.reshape(21, 3)
        hand_valid[index, side] = True
    translations = np.full((n, 3), np.nan)
    quaternions = np.full((n, 4), np.nan)
    pose_valid = np.zeros(n, dtype=bool)
    seen_poses = np.zeros(n, dtype=bool)
    world_frame = None
    for log_ns, message in poses:
        index = _frame_index(times, log_ns)
        if seen_poses[index]:
            raise DataError("Duplicate pose sample for camera frame")
        seen_poses[index] = True
        transforms = [t for t in message.transforms if str(t.child_frame_id) == camera_id]
        if len(transforms) != 1:
            raise DataError("Expected exactly one camera-to-world VIO transform")
        transform = transforms[0]
        _check_timestamp(transform, log_ns, POSE_TOPIC, metadata)
        parent = str(transform.parent_frame_id)
        if not parent or parent == camera_id or (world_frame is not None and parent != world_frame):
            raise DataError("Missing or changing VIO world coordinate frame")
        world_frame = parent
        t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        q = np.array([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z])
        norm = np.linalg.norm(q)
        if not np.isfinite(t).all() or not np.isfinite(q).all() or abs(norm - 1) > 1e-3:
            continue
        translations[index], quaternions[index] = t, q / norm
        pose_valid[index] = True
    subtasks = [""] * n
    subtask_seen = set()
    for timestamp, text in subtask_rows:
        index = _frame_index(times, timestamp)
        if index in subtask_seen:
            raise DataError("Duplicate subtask annotation for frame")
        subtask_seen.add(index)
        subtasks[index] = text
    if annotations and any(item != annotations[0] for item in annotations[1:]):
        raise DataError("Conflicting task annotations")
    annotation = annotations[0] if annotations else {}
    supplemental = metadata["source_metadata"].get("supplemental", {})
    if not isinstance(supplemental, dict):
        supplemental = {}
    metadata.update(
        camera_frame_id=camera_id, vio_parent_frame_id=world_frame,
        vio_transform_direction="camera_to_world", skeleton_sources=sorted(sources),
        task_annotation=annotation, missing_hand_samples=(~seen_hands).sum(axis=0).tolist(),
        missing_pose_samples=int((~seen_poses).sum()),
        missing_payload_timestamps=dict(metadata["missing_payload_timestamps"]),
        h264_alignment=packet_guard.summary(),
    )
    identity = _source_identity(path, n, times, fps, annotation)
    metadata["recording_time_origin_ns"] = identity.pop("recording_time_origin_ns")
    metadata["source_identity_reason"] = identity.pop("source_identity_reason")
    metadata["directory_end_convention"] = identity.pop("directory_end_convention", None)
    result = dict(
        source_path=path, **identity, task=annotation.get("task_name") or supplemental.get("task_description") or "unknown",
        skill_group=supplemental.get("skill_group", "unknown"), metadata=metadata,
        n=n, fps=fps, width=width, height=height, times_ns=times, K=K, D=D,
        positions_m=positions, hand_valid=hand_valid, vio_t=translations,
        vio_q_wxyz=quaternions, pose_valid=pose_valid, subtasks=subtasks,
    )
    if include_video:
        result["video_packets"] = videos
    return result


def project_hands(ep):
    """Return equidistant projections and positive-depth validity, without clipping.

    Positive depth is not evidence of in-frame or unoccluded RGB visibility.
    """
    points = np.asarray(ep["positions_m"], dtype=float)
    n = int(ep["n"])
    if points.shape != (n, 2, 21, 3) or np.shape(ep["hand_valid"]) != (n, 2):
        raise DataError("Incorrect aligned hand array shape")
    valid = (np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-6)
             & np.asarray(ep["hand_valid"], dtype=bool)[..., None])
    safe_z = np.where(valid, points[..., 2], 1.0)
    x = np.where(valid, points[..., 0] / safe_z, 0.0)
    y = np.where(valid, points[..., 1] / safe_z, 0.0)
    radius = np.hypot(x, y)
    theta = np.arctan(radius)
    theta2 = theta * theta
    k1, k2, k3, k4 = np.asarray(ep["D"], dtype=float)
    distorted = theta * (1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
    scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
    K = np.asarray(ep["K"], dtype=float)
    uv = np.stack((K[0, 0] * x * scale + K[0, 2], K[1, 1] * y * scale + K[1, 2]), axis=-1)
    valid &= np.isfinite(uv).all(axis=-1)
    uv[~valid] = np.nan
    return uv, valid


def _ffmpeg_executable():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        raise DataError("FFmpeg unavailable; install imageio-ffmpeg in the runtime")


def iter_video_packets(path):
    """Yield encoded camera packets in the same MCAP order used for RGB decoding."""
    previous = None
    guard = _H264PacketGuard()
    for _, timestamp, message in _messages(path, [VIDEO_TOPIC]):
        if previous is not None and timestamp <= previous:
            raise DataError("Duplicate or nonmonotonic video packet timestamps")
        if str(message.format).lower() not in ("h264", "h.264") or not message.data:
            raise DataError("Expected nonempty H264 camera packets")
        payload_time = _timestamp_ns(message)
        if payload_time is not None and abs(payload_time - timestamp) > TIME_TOLERANCE_NS:
            raise DataError("Video packet payload/log timestamp disagreement")
        previous = timestamp
        guard.check(message.data)
        yield bytes(message.data)


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return data


def iter_video_frames(path, ep, scale_width=None):
    """Decode to one RGB frame at a time, draining stderr and feeding concurrently.

    Exhaust the iterator to receive decoder and exact-frame-count validation.
    Closing it early terminates its private FFmpeg process and closes source I/O.
    No source files, temporary encoded files or full-video RGB buffers are written.
    """
    width, height = int(ep["width"]), int(ep["height"])
    if scale_width is not None:
        if isinstance(scale_width, bool) or int(scale_width) != scale_width or scale_width < 1:
            raise DataError("scale_width must be a positive integer")
        output_width = min(width, int(scale_width))
        output_height = max(1, round(height * output_width / width))
    else:
        output_width, output_height = width, height
    command = [_ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-f", "h264", "-i", "pipe:0", "-map", "0:v:0", "-an", "-vsync", "0"]
    if (output_width, output_height) != (width, height):
        command.extend(["-vf", f"scale={output_width}:{output_height}:flags=area"])
    command.extend(["-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    errors = []
    stderr = collections.deque(maxlen=128)
    stopped = threading.Event()

    def feed():
        count = 0
        guard = _H264PacketGuard()
        try:
            for _, timestamp, message in _messages(path, [VIDEO_TOPIC]):
                if stopped.is_set():
                    return
                if count >= ep["n"] or abs(timestamp - int(ep["times_ns"][count])) > TIME_TOLERANCE_NS:
                    raise DataError("Video source timeline changed since metadata read")
                guard.check(message.data)
                if guard.dimensions != (width, height):
                    raise DataError("Video source dimensions changed since metadata read")
                packet = memoryview(message.data)
                while packet and not stopped.is_set():
                    written = process.stdin.write(packet)
                    if not written:
                        raise BrokenPipeError("Video decoder stopped accepting input")
                    packet = packet[written:]
                count += 1
            if count != ep["n"]:
                raise DataError(f"Video packet count changed: {count} != {ep['n']}")
        except Exception as exc:
            if not stopped.is_set():
                errors.append(exc)
        finally:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

    def drain_stderr():
        try:
            while True:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                stderr.append(chunk)
        except (OSError, ValueError):
            return

    feeder = threading.Thread(target=feed, daemon=True, name="handclip-h264-feed")
    drainer = threading.Thread(target=drain_stderr, daemon=True, name="handclip-ffmpeg-stderr")
    feeder.start()
    drainer.start()
    count = 0
    try:
        frame_bytes = output_width * output_height * 3
        while True:
            raw = _read_exact(process.stdout, frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise DataError("Decoder emitted a truncated RGB frame")
            count += 1
            if count > ep["n"]:
                raise DataError("Decoder emitted more frames than the source timeline")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(output_height, output_width, 3)
        feeder.join(timeout=30)
        if feeder.is_alive():
            raise DataError("Video feeder did not finish after decoder EOF")
        returncode = process.wait(timeout=30)
        drainer.join(timeout=5)
        detail = b"".join(stderr).decode("utf-8", errors="replace")[-4000:]
        if errors:
            raise DataError(f"Video input failed: {errors[0]}; {detail}") from errors[0]
        if returncode != 0 or detail.strip():
            raise DataError(f"FFmpeg decode failed ({returncode}): {detail}")
        if count != ep["n"]:
            raise DataError(f"Decoded frame count mismatch: {count} != {ep['n']}")
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        feeder.join(timeout=5)
        drainer.join(timeout=5)
