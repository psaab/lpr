"""Processing pipeline - orchestrates detection, OCR, and output."""

import logging
import time
from pathlib import Path

from .config import Config
from .detector import PlateDetector
from .output import JSONOutputWriter
from .reader import PlateReader
from .stream import StreamReader

logger = logging.getLogger("lpr.pipeline")


class Pipeline:
    """Orchestrates the full license plate recognition pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self._device = config.resolve_device()
        self._seen_plates: dict[str, float] = {}  # plate_text -> last_seen_time
        self._stats = {"frames": 0, "detections": 0, "plates_read": 0}

    def run(self) -> None:
        """Run the pipeline on the configured video source."""
        logger.info("Starting LPR pipeline on source: %s", self.config.source)

        stream = StreamReader(
            source=self.config.source,
            frame_skip=self.config.frame_skip,
            reconnect_delay=self.config.reconnect_delay,
            max_reconnects=self.config.max_reconnects,
        )

        detector = PlateDetector(
            device=self._device,
            confidence=self.config.confidence_threshold,
        )
        detector.load_model()

        reader = PlateReader(device=self._device)
        reader.load_model()

        writer = JSONOutputWriter(Path(self.config.output_path))

        logger.info("All models loaded. Processing frames...")

        try:
            with writer:
                for frame in stream.frames():
                    self._stats["frames"] += 1
                    self._process_frame(frame, detector, reader, writer)

                    if self._stats["frames"] % 100 == 0:
                        logger.info(
                            "Progress: %d frames, %d detections, %d plates read",
                            self._stats["frames"],
                            self._stats["detections"],
                            self._stats["plates_read"],
                        )
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            stream.close()
            logger.info(
                "Pipeline finished. Stats: %d frames, %d detections, %d plates",
                self._stats["frames"],
                self._stats["detections"],
                self._stats["plates_read"],
            )

    def _process_frame(self, frame, detector, reader, writer):
        """Process a single frame through detection and OCR."""
        from .stream import Frame

        detections = detector.detect(frame.image)
        self._stats["detections"] += len(detections)

        for detection in detections:
            plate = reader.read(detection.plate_image)
            if plate is None or not plate.text:
                continue

            # Deduplication: skip if we saw this plate recently
            now = time.time()
            last_seen = self._seen_plates.get(plate.text)
            if last_seen and (now - last_seen) < self.config.dedup_seconds:
                continue

            self._seen_plates[plate.text] = now
            self._stats["plates_read"] += 1

            record = writer.make_record(
                plate_text=plate.text,
                confidence=plate.confidence,
                timestamp=frame.timestamp,
                frame_number=frame.frame_number,
                bbox=detection.bbox,
                source=frame.source,
            )
            writer.write(record)

            logger.info(
                "Plate detected: %s (confidence: %.2f) at frame %d",
                plate.text, plate.confidence, frame.frame_number,
            )

        # Prune old entries from seen_plates
        now = time.time()
        cutoff = now - self.config.dedup_seconds * 3
        self._seen_plates = {
            k: v for k, v in self._seen_plates.items() if v > cutoff
        }
