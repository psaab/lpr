"""JSON output writer for plate detections."""

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("lpr.output")


@dataclass
class DetectionRecord:
    """A single plate detection record."""

    plate_text: str
    confidence: float
    timestamp: float
    frame_number: int
    bbox: tuple[int, int, int, int]
    source: str
    detected_at: str
    consensus_count: int = 0
    consensus_confidence: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(d["bbox"])
        return d


class JSONOutputWriter:
    """Writes detection records as JSONL (one JSON object per line)."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._lock = threading.Lock()
        self._file = None
        self._count = 0

    def open(self) -> None:
        """Open the output file for writing."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")
        logger.info("Writing detections to %s", self.output_path)

    def write(self, record: DetectionRecord) -> None:
        """Write a detection record as a JSON line."""
        if self._file is None:
            self.open()

        line = json.dumps(record.to_dict(), ensure_ascii=False)

        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            self._count += 1

        logger.debug("Wrote detection #%d: %s", self._count, record.plate_text)

    def close(self) -> None:
        """Close the output file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info("Closed output file. Total detections written: %d", self._count)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def make_record(
        plate_text: str,
        confidence: float,
        timestamp: float,
        frame_number: int,
        bbox: tuple[int, int, int, int],
        source: str,
        consensus_count: int = 0,
        consensus_confidence: float = 0.0,
    ) -> DetectionRecord:
        return DetectionRecord(
            plate_text=plate_text,
            confidence=confidence,
            timestamp=timestamp,
            frame_number=frame_number,
            bbox=bbox,
            source=source,
            detected_at=datetime.now(timezone.utc).isoformat(),
            consensus_count=consensus_count,
            consensus_confidence=consensus_confidence,
        )
