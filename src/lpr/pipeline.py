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
from .vehicle import VehicleClassifier

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
        self._vehicle_classifier: VehicleClassifier | None = None
        if config.vehicle_attrs:
            self._vehicle_classifier = VehicleClassifier(
                device=self._device,
                make_model_path=config.vehicle_make_model_path,
                make_model_labels_path=config.vehicle_make_model_labels,
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
            tls_verify=self.config.tls_verify,
            tls_ca_file=(
                str(self.config.tls_ca_file) if self.config.tls_ca_file else None
            ),
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

        if self._vehicle_classifier is not None:
            self._vehicle_classifier.load_model()

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

            # Vehicle attribute detection
            v_color = None
            v_type = None
            v_mm = None
            v_color_conf = 0.0
            v_type_conf = 0.0
            v_mm_conf = 0.0
            if self._vehicle_classifier is not None:
                attrs = self._vehicle_classifier.classify(
                    frame.image, detection.bbox
                )
                if attrs is not None:
                    v_color = attrs.color
                    v_type = attrs.vehicle_type
                    v_mm = attrs.make_model
                    v_color_conf = attrs.color_confidence
                    v_type_conf = attrs.type_confidence
                    v_mm_conf = attrs.make_model_confidence

            reading = PlateReading(
                text=plate.text,
                confidence=plate.confidence,
                bbox=detection.bbox,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                source=frame.source,
                vehicle_color=v_color,
                vehicle_type=v_type,
                vehicle_make_model=v_mm,
                vehicle_color_confidence=v_color_conf,
                vehicle_type_confidence=v_type_conf,
                vehicle_make_model_confidence=v_mm_conf,
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
                vehicle_color=result.vehicle_color,
                vehicle_type=result.vehicle_type,
                vehicle_make_model=result.vehicle_make_model,
            )
            writer.write(record)

            vehicle_info = ""
            if result.vehicle_color or result.vehicle_type:
                parts = [
                    result.vehicle_color or "",
                    result.vehicle_type or "",
                ]
                vehicle_info = " vehicle: " + " ".join(p for p in parts if p)
                if result.vehicle_make_model:
                    vehicle_info += f" ({result.vehicle_make_model})"

            logger.info(
                "Plate consensus: %s (confidence: %.2f, readings: %d, "
                "consensus: %.2f) at frame %d%s",
                result.text,
                plate.confidence,
                result.reading_count,
                result.consensus_confidence,
                result.frame_number,
                vehicle_info,
            )
