"""Video stream handler - reads frames from files or RTSP streams."""

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger("lpr.stream")

# Max output height for GPU downscale — only scale if input exceeds this
_MAX_OUTPUT_HEIGHT = 1080


@dataclass
class Frame:
    """A video frame with metadata."""

    image: np.ndarray
    frame_number: int
    timestamp: float
    source: str


def _probe_video(source: str) -> dict | None:
    """Use ffprobe to get video stream metadata."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v:0",
                source,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None

        s = streams[0]
        # Parse fps from r_frame_rate (e.g. "30000/1001")
        fps_str = s.get("r_frame_rate", "0/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 0.0

        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": fps,
            "codec": s.get("codec_name", "unknown"),
        }
    except Exception as e:
        logger.debug("ffprobe failed: %s", e)
        return None


def _check_ffmpeg_hwaccel(ffmpeg: str) -> list[str]:
    """Check which HW acceleration methods ffmpeg supports."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        methods = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line not in ("Hardware acceleration methods:",):
                methods.append(line)
        return methods
    except Exception:
        return []


class StreamReader:
    """Reads frames from video files or RTSP streams."""

    def __init__(
        self,
        source: str,
        frame_skip: int = 5,
        reconnect_delay: float = 5.0,
        max_reconnects: int = 10,
    ):
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._is_stream = source.startswith("rtsp://") or source.startswith("http")
        self._cap: cv2.VideoCapture | None = None
        self._proc: subprocess.Popen | None = None
        self._frame_size: int = 0
        self._out_width: int = 0
        self._out_height: int = 0
        self._src_width: int = 0
        self._src_height: int = 0
        self._fps: float = 0.0
        self._use_ffmpeg: bool = False

    def _build_ffmpeg_strategies(
        self, ffmpeg: str, hwaccels: list[str], src_w: int, src_h: int
    ) -> list[tuple[list[str], int, int, str]]:
        """Build a list of ffmpeg command strategies to try, best first.

        Returns list of (cmd_args, output_width, output_height, description).
        """
        strategies = []

        # Compute downscaled dimensions (even values required for NV12)
        need_scale = src_h > _MAX_OUTPUT_HEIGHT
        if need_scale:
            out_h = _MAX_OUTPUT_HEIGHT
            out_w = int(src_w * (out_h / src_h))
            out_w = out_w + (out_w % 2)  # ensure even
        else:
            out_w, out_h = src_w, src_h

        nv12_out = ["-f", "rawvideo", "-pix_fmt", "nv12", "-v", "error", "-"]

        if "cuda" in hwaccels:
            if need_scale:
                # Strategy 1: CUDA decode + GPU scale + NV12 output
                vf = f"scale_cuda=w={out_w}:h={out_h},hwdownload,format=nv12"
                cmd = [
                    ffmpeg,
                    "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda",
                    "-i", self.source,
                    "-vf", vf,
                ] + nv12_out
                strategies.append(
                    (cmd, out_w, out_h, f"HW decode + GPU scale {src_w}x{src_h}->{out_w}x{out_h}")
                )

            # Strategy 2: CUDA decode + NV12 output (no CPU swscale)
            cmd = [
                ffmpeg,
                "-hwaccel", "cuda",
                "-i", self.source,
            ] + nv12_out
            strategies.append(
                (cmd, src_w, src_h, f"HW decode (cuda/nvdec)")
            )

        if "vaapi" in hwaccels:
            # Strategy 3: VAAPI decode + NV12 output
            cmd = [
                ffmpeg,
                "-hwaccel", "vaapi",
                "-i", self.source,
            ] + nv12_out
            strategies.append(
                (cmd, src_w, src_h, f"HW decode (vaapi)")
            )

        return strategies

    def _try_ffmpeg_strategy(
        self, cmd: list[str], out_w: int, out_h: int
    ) -> bool:
        """Try an ffmpeg command, return True if it produces a valid frame."""
        frame_size = out_w * out_h * 3 // 2  # NV12: 1.5 bytes/pixel

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size * 4,
            )
            data = proc.stdout.read(frame_size)
            # Clean up test process
            proc.stdout.close()
            proc.stderr.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

            return len(data) == frame_size

        except Exception:
            return False

    def _open_ffmpeg(self) -> bool:
        """Try to open source with ffmpeg HW-accelerated decode."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return False

        info = _probe_video(self.source)
        if info is None or info["width"] == 0:
            return False

        self._src_width = info["width"]
        self._src_height = info["height"]
        self._fps = info["fps"]

        hwaccels = _check_ffmpeg_hwaccel(ffmpeg)
        strategies = self._build_ffmpeg_strategies(
            ffmpeg, hwaccels, self._src_width, self._src_height
        )

        if not strategies:
            return False

        for cmd, out_w, out_h, desc in strategies:
            if not self._try_ffmpeg_strategy(cmd, out_w, out_h):
                logger.debug("ffmpeg strategy failed: %s", desc)
                continue

            # Strategy works — open for real
            self._out_width = out_w
            self._out_height = out_h
            self._frame_size = out_w * out_h * 3 // 2

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 4,
            )
            self._use_ffmpeg = True

            logger.info(
                "Opened source: %s (%dx%d @ %.1f fps, %s via ffmpeg, NV12 output)",
                self.source, out_w, out_h, self._fps, desc,
            )
            return True

        return False

    def _close_ffmpeg(self) -> None:
        """Terminate ffmpeg subprocess."""
        if self._proc is not None:
            try:
                self._proc.stdout.close()
                self._proc.stderr.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _open_cv2(self) -> cv2.VideoCapture:
        """Open source with OpenCV VideoCapture (software decode)."""
        if not self._is_stream:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self._out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 0

        logger.info(
            "Opened source: %s (%dx%d @ %.1f fps, software decode via OpenCV)",
            self.source, self._out_width, self._out_height, self._fps,
        )
        return cap

    def _open(self) -> cv2.VideoCapture | None:
        """Open the video source, trying HW decode first."""
        if not self._is_stream:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")

        if self._open_ffmpeg():
            return None  # using ffmpeg subprocess, not cv2

        return self._open_cv2()

    def frames(self) -> Generator[Frame, None, None]:
        """Yield frames from the video source."""
        reconnects = 0
        frame_number = 0

        while True:
            try:
                self._use_ffmpeg = False
                self._cap = self._open()
                reconnects = 0

                if self._use_ffmpeg:
                    yield from self._frames_ffmpeg(frame_number)
                else:
                    for frame in self._frames_cv2(frame_number):
                        yield frame
                    frame_number = 0  # cv2 reached end
                    return

            except RuntimeError:
                if not self._is_stream:
                    raise
            finally:
                self.close()

            if not self._is_stream:
                return

            reconnects += 1
            if reconnects > self.max_reconnects:
                logger.error("Max reconnection attempts reached (%d)", self.max_reconnects)
                return

            logger.info(
                "Reconnecting in %.1fs (attempt %d/%d)",
                self.reconnect_delay, reconnects, self.max_reconnects,
            )
            time.sleep(self.reconnect_delay)

    def _frames_ffmpeg(self, start_frame: int) -> Generator[Frame, None, None]:
        """Yield frames from ffmpeg subprocess (NV12 input, BGR output)."""
        frame_number = start_frame
        w, h = self._out_width, self._out_height
        nv12_h = h * 3 // 2  # NV12 has 1.5x height

        while self._proc is not None:
            data = self._proc.stdout.read(self._frame_size)
            if len(data) != self._frame_size:
                if self._is_stream:
                    logger.warning("Stream read failed, will reconnect")
                else:
                    logger.info("End of video file reached")
                return

            frame_number += 1
            if frame_number % self.frame_skip != 0:
                continue

            # Reshape as NV12 and convert to BGR
            nv12 = np.frombuffer(data, dtype=np.uint8).reshape((nv12_h, w))
            image = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

            timestamp = frame_number / self._fps if self._fps > 0 else 0.0

            yield Frame(
                image=image,
                frame_number=frame_number,
                timestamp=timestamp,
                source=self.source,
            )

    def _frames_cv2(self, start_frame: int) -> Generator[Frame, None, None]:
        """Yield frames from OpenCV VideoCapture."""
        frame_number = start_frame

        while True:
            ret, image = self._cap.read()
            if not ret:
                if self._is_stream:
                    logger.warning("Stream read failed, will reconnect")
                    return
                logger.info("End of video file reached")
                return

            frame_number += 1
            if frame_number % self.frame_skip != 0:
                continue

            timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield Frame(
                image=image,
                frame_number=frame_number,
                timestamp=timestamp,
                source=self.source,
            )

    def close(self) -> None:
        """Release resources."""
        self._close_ffmpeg()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
