"""Processing pipeline - orchestrates detection, OCR, consensus, and output."""

import logging
import time
from pathlib import Path

from .config import Config
from .consensus import PlateConsensus, PlateReading
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
        self._consensus = PlateConsensus(
            min_readings=config.min_readings,
            consensus_threshold=config.consensus_threshold,
            track_timeout=config.track_timeout,
        )
        self._stats = {
            "frames": 0,
            "detections": 0,
            "plates_read": 0,
            "consensus_emitted": 0,
        }

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
            detection_model=self.config.detection_model,
            use_legacy=self.config.use_legacy_detector,
        )
        detector.load_model()

        reader = PlateReader(
            device=self._device,
            ocr_engine=self.config.ocr_engine,
        )
        reader.load_model()

        writer = JSONOutputWriter(Path(self.config.output_path))

        logger.info("All models loaded. Processing frames...")

        try:
            with writer:
                for frame in stream.frames():
                    self._stats["frames"] += 1
                    self._process_frame(frame, detector, reader, writer)

                    if self._stats["frames"] % 50 == 0:
                        self._consensus.prune_stale_tracks(time.time())

                    if self._stats["frames"] % 100 == 0:
                        logger.info(
                            "Progress: %d frames, %d detections, "
                            "%d plates read, %d consensus emitted",
                            self._stats["frames"],
                            self._stats["detections"],
                            self._stats["plates_read"],
                            self._stats["consensus_emitted"],
                        )
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            stream.close()
            logger.info(
                "Pipeline finished. Stats: %d frames, %d detections, "
                "%d plates read, %d consensus emitted",
                self._stats["frames"],
                self._stats["detections"],
                self._stats["plates_read"],
                self._stats["consensus_emitted"],
            )

    def _process_frame(self, frame, detector, reader, writer):
        """Process a single frame through detection, OCR, and consensus."""
        from .stream import Frame

        detections = detector.detect(frame.image)
        self._stats["detections"] += len(detections)

        for detection in detections:
            plate = reader.read(detection.plate_image)
            if plate is None or not plate.text:
                continue

            self._stats["plates_read"] += 1

            reading = PlateReading(
                text=plate.text,
                confidence=plate.confidence,
                bbox=detection.bbox,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                source=frame.source,
            )

            result = self._consensus.add_reading(reading)
            if result is None:
                continue

            self._stats["consensus_emitted"] += 1

            record = writer.make_record(
                plate_text=result.text,
                confidence=plate.confidence,
                timestamp=result.last_seen,
                frame_number=result.frame_number,
                bbox=result.bbox,
                source=result.source,
                consensus_count=result.reading_count,
                consensus_confidence=result.consensus_confidence,
            )
            writer.write(record)

            logger.info(
                "Plate consensus: %s (confidence: %.2f, readings: %d, "
                "consensus: %.2f) at frame %d",
                result.text,
                plate.confidence,
                result.reading_count,
                result.consensus_confidence,
                result.frame_number,
            )
