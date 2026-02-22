"""Processing pipeline - orchestrates detection, OCR, consensus, and output."""

import logging
import threading
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
        self._vehicle_classifier: VehicleClassifier | None = None
        if config.vehicle_attrs:
            self._vehicle_classifier = VehicleClassifier(
                device=self._device,
                make_model_path=config.vehicle_make_model_path,
                make_model_labels_path=config.vehicle_make_model_labels,
            )
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Run the pipeline on all configured video sources."""
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

        sources = self.config.sources

        if len(sources) == 1:
            # Single source — run directly, no threading overhead
            try:
                with writer:
                    self._run_source(sources[0], detector, reader, writer)
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted by user")
        else:
            # Multiple sources — one thread per source
            threads: list[threading.Thread] = []
            for source in sources:
                stem = Path(source).stem if "://" not in source else source
                t = threading.Thread(
                    target=self._run_source,
                    args=(source, detector, reader, writer),
                    name=f"lpr-{stem}",
                    daemon=True,
                )
                threads.append(t)

            try:
                with writer:
                    for t in threads:
                        t.start()
                    # Wait for all threads, checking for stop periodically
                    while any(t.is_alive() for t in threads):
                        for t in threads:
                            t.join(timeout=0.5)
                            if self._stop_event.is_set():
                                break
                        if self._stop_event.is_set():
                            break
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted by user, stopping all sources...")
                self._stop_event.set()
                for t in threads:
                    t.join(timeout=5.0)

    def _run_source(self, source: str, detector, reader, writer) -> None:
        """Run the pipeline loop for a single source."""
        stem = Path(source).stem if "://" not in source else source
        log = logging.getLogger(f"lpr.pipeline.{stem}")
        log.info("Starting on source: %s", source)

        consensus = PlateConsensus(
            min_readings=self.config.min_readings,
            consensus_threshold=self.config.consensus_threshold,
            track_timeout=self.config.track_timeout,
        )
        stats = {
            "frames": 0,
            "detections": 0,
            "plates_read": 0,
            "consensus_emitted": 0,
        }

        stream = StreamReader(
            source=source,
            frame_skip=self.config.frame_skip,
            reconnect_delay=self.config.reconnect_delay,
            max_reconnects=self.config.max_reconnects,
            tls_verify=self.config.tls_verify,
            tls_ca_file=(
                str(self.config.tls_ca_file) if self.config.tls_ca_file else None
            ),
            stall_timeout=self.config.stall_timeout,
        )

        try:
            for frame in stream.frames():
                if self._stop_event.is_set():
                    break

                stats["frames"] += 1
                self._process_frame(
                    frame, detector, reader, writer, consensus, stats, log,
                )

                if stats["frames"] % 50 == 0:
                    consensus.prune_stale_tracks(time.time())

                if stats["frames"] % 100 == 0:
                    log.info(
                        "Progress: %d frames, %d detections, "
                        "%d plates read, %d consensus emitted",
                        stats["frames"],
                        stats["detections"],
                        stats["plates_read"],
                        stats["consensus_emitted"],
                    )
        except KeyboardInterrupt:
            # Only happens in single-source (main thread) mode; re-raise
            raise
        except Exception:
            log.exception("Source %s failed", source)
        finally:
            stream.close()
            log.info(
                "Source finished. Stats: %d frames, %d detections, "
                "%d plates read, %d consensus emitted",
                stats["frames"],
                stats["detections"],
                stats["plates_read"],
                stats["consensus_emitted"],
            )

    def _process_frame(self, frame, detector, reader, writer, consensus, stats, log):
        """Process a single frame through detection, OCR, and consensus."""
        detections = detector.detect(frame.image)
        stats["detections"] += len(detections)

        for detection in detections:
            plate = reader.read(detection.plate_image)
            if plate is None or not plate.text:
                continue

            stats["plates_read"] += 1

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

            result = consensus.add_reading(reading)
            if result is None:
                continue

            stats["consensus_emitted"] += 1

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

            if self.config.snapshot_path:
                writer.save_snapshot(
                    frame=frame.image,
                    plate_text=result.text,
                    timestamp=result.last_seen,
                    frame_number=result.frame_number,
                    source=result.source,
                    path_template=self.config.snapshot_path,
                )

            vehicle_info = ""
            if result.vehicle_color or result.vehicle_type:
                parts = [
                    result.vehicle_color or "",
                    result.vehicle_type or "",
                ]
                vehicle_info = " vehicle: " + " ".join(p for p in parts if p)
                if result.vehicle_make_model:
                    vehicle_info += f" ({result.vehicle_make_model})"

            log.info(
                "Plate consensus: %s (confidence: %.2f, readings: %d, "
                "consensus: %.2f) at frame %d%s",
                result.text,
                plate.confidence,
                result.reading_count,
                result.consensus_confidence,
                result.frame_number,
                vehicle_info,
            )
