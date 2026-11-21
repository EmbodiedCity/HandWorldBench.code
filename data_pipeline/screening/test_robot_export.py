import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from robot_io import FRAME_MAPPING, case_id, encode, sha, verify_clip_receipt


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class RobotExportTest(unittest.TestCase):
    def test_irregular_video_timestamps_preserve_every_frame_of_the_trimmed_prefix(self):
        width, height = 16, 16
        pixels_per_frame = width * height
        raw = b"".join(bytes([20 + 2 * index]) * pixels_per_frame for index in range(82))
        timestamps = (
            "settb=expr=1/1000,setpts='if(eq(N,0),0,if(eq(N,1),55,if(eq(N,2),70,"
            "if(eq(N,3),75,if(eq(N,4),118,if(eq(N,5),155,"
            "if(eq(N,81),10000,round(N*1000/30))))))))'"
        )

        def decoded_labels(path):
            decoded = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ])
            return list(decoded[::pixels_per_frame])

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-filter_threads", "1",
                "-f", "rawvideo", "-pix_fmt", "gray", "-video_size", f"{width}x{height}",
                "-framerate", "30", "-i", "-", "-vf", timestamps, "-enc_time_base", "1:1000",
                "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "18", "-preset",
                "fast", "-threads", "1", "-pix_fmt", "yuv420p", str(source),
            ], input=raw, check=True)
            original = decoded_labels(source)
            self.assertEqual(len(original), 82)
            self.assertEqual(len(set(original)), 82)
            for rate in (30, "30000/1001"):
                with self.subTest(rate=rate):
                    output = Path(directory) / "clip.mp4"
                    receipt = encode(source, output, 81, rate)
                    self.assertEqual(receipt["frame_mapping"], FRAME_MAPPING)
                    self.assertEqual(receipt["frame_rate"], str(rate))
                    self.assertEqual(decoded_labels(output), original[:81])


class ResumeReceiptTest(unittest.TestCase):
    def test_missing_or_changed_secondary_assets_and_legacy_receipts_are_rejected(self):
        source = {"episode_uid": "source", "frames": 82, "fps": 30}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_frame = root / "first_frame.png"
            first_frame.write_bytes(b"first frame")
            views = {}
            for name in ("front", "wrist_left", "wrist_right"):
                path = root / f"{name}.mp4"
                path.write_bytes(name.encode())
                views[name] = dict(path=str(path), sha256=sha(path), decoded_frames=81,
                                   frame_mapping=FRAME_MAPPING, frame_rate="30")
            primary = views["front"]
            record = dict(
                id=case_id("handson", source), domain="handson", source_identity="source",
                source_frames=[0, 81], frames=81, source_raw_frames=82, fps=30, videos=views,
                video=primary["path"], video_sha=primary["sha256"],
                first_frame=str(first_frame), first_frame_sha256=sha(first_frame),
            )
            verify_clip_receipt(record, source, "handson")

            del primary["frame_mapping"]
            with self.assertRaisesRegex(ValueError, "frame-mapping policy"):
                verify_clip_receipt(record, source, "handson")
            primary["frame_mapping"] = FRAME_MAPPING

            wrist = Path(views["wrist_left"]["path"])
            wrist.write_bytes(b"corrupted view")
            with self.assertRaisesRegex(ValueError, "video SHA-256"):
                verify_clip_receipt(record, source, "handson")
            wrist.write_bytes(b"wrist_left")

            first_frame.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_clip_receipt(record, source, "handson")


if __name__ == "__main__":
    unittest.main()
