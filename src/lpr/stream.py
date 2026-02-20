"""Video stream handler - reads frames from files or RTSP streams."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger("lpr.stream")


@dataclass
class Frame:
    """A video frame with metadata."""

    image: np.ndarray
    frame_number: int
    timestamp: float
    source: str


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

    def _open(self) -> cv2.VideoCapture:
        """Open the video source."""
        if not self._is_stream:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        logger.info(
            "Opened source: %s (%dx%d @ %.1f fps)",
            self.source, width, height, fps,
        )
        return cap

    def frames(self) -> Generator[Frame, None, None]:
        """Yield frames from the video source."""
        reconnects = 0
        frame_number = 0

        while True:
            try:
                self._cap = self._open()
                reconnects = 0

                while True:
                    ret, image = self._cap.read()
                    if not ret:
                        if self._is_stream:
                            logger.warning("Stream read failed, will reconnect")
                            break
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

            except RuntimeError:
                if not self._is_stream:
                    raise
            finally:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None

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

    def close(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
