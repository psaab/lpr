"""Multi-frame consensus voting for license plate readings."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class PlateReading:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    frame_number: int
    timestamp: float
    source: str
    vehicle_color: str | None = None
    vehicle_type: str | None = None
    vehicle_make_model: str | None = None
    vehicle_color_confidence: float = 0.0
    vehicle_type_confidence: float = 0.0
    vehicle_make_model_confidence: float = 0.0


@dataclass
class ConsensusResult:
    text: str
    consensus_confidence: float
    reading_count: int
    first_seen: float
    last_seen: float
    bbox: tuple[int, int, int, int]
    source: str
    frame_number: int
    vehicle_color: str | None = None
    vehicle_type: str | None = None
    vehicle_make_model: str | None = None


@dataclass
class PlateTrack:
    readings: list[PlateReading] = field(default_factory=list)
    emitted: bool = False

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.readings[-1].bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def _iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Compute intersection over union of two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def _centroid_distance(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> float:
    """Compute euclidean distance between the centers of two bounding boxes."""
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


class PlateConsensus:
    """Accumulates plate readings across frames and emits a consensus result."""

    def __init__(
        self,
        min_readings: int = 3,
        consensus_threshold: float = 0.6,
        track_timeout: float = 5.0,
    ) -> None:
        self.min_readings = min_readings
        self.consensus_threshold = consensus_threshold
        self.track_timeout = track_timeout
        self._tracks: list[PlateTrack] = []

    def add_reading(self, reading: PlateReading) -> ConsensusResult | None:
        """Add a reading, returning a ConsensusResult if consensus is reached."""
        track = self._find_matching_track(reading)
        if track is None:
            track = PlateTrack()
            self._tracks.append(track)

        track.readings.append(reading)

        if len(track.readings) >= self.min_readings and not track.emitted:
            return self._compute_consensus(track)
        return None

    def _find_matching_track(self, reading: PlateReading) -> PlateTrack | None:
        """Find a track matching this reading by IoU, centroid, or text similarity."""
        best_iou = 0.0
        best_track: PlateTrack | None = None

        for track in self._tracks:
            last_bbox = track.readings[-1].bbox
            iou = _iou(reading.bbox, last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_iou >= 0.3:
            return best_track

        # Centroid distance fallback — scale threshold by bbox diagonal
        # so it works on both 720p and 4K frames
        best_dist = float("inf")
        best_track = None

        for track in self._tracks:
            last_bbox = track.readings[-1].bbox
            diag = math.sqrt(
                (last_bbox[2] - last_bbox[0]) ** 2
                + (last_bbox[3] - last_bbox[1]) ** 2
            )
            # Allow centroid to move up to 3x the plate diagonal between frames
            threshold = max(100.0, diag * 3.0)
            dist = _centroid_distance(reading.bbox, last_bbox)
            if dist < best_dist and dist <= threshold:
                best_dist = dist
                best_track = track

        if best_track is not None:
            return best_track

        # Text similarity fallback — if the plate text closely matches
        # an existing track, it's likely the same plate even if it moved far
        for track in self._tracks:
            if track.emitted:
                continue
            last_text = track.readings[-1].text
            if len(reading.text) == len(last_text):
                mismatches = sum(
                    a != b for a, b in zip(reading.text, last_text)
                )
                if mismatches <= 1:
                    return track

        return None

    def _compute_consensus(self, track: PlateTrack) -> ConsensusResult | None:
        """Character-level weighted majority vote over readings in a track."""
        readings = track.readings

        # Filter to dominant text length
        length_counts: Counter[int] = Counter(len(r.text) for r in readings)
        dominant_length = length_counts.most_common(1)[0][0]
        filtered = [r for r in readings if len(r.text) == dominant_length]

        if not filtered:
            return None

        # Weighted majority vote per character position
        consensus_chars: list[str] = []
        position_agreements: list[float] = []

        for pos in range(dominant_length):
            char_weights: Counter[str] = Counter()
            total_weight = 0.0

            for r in filtered:
                char_weights[r.text[pos]] += r.confidence
                total_weight += r.confidence

            winner, winner_weight = char_weights.most_common(1)[0]
            consensus_chars.append(winner)
            position_agreements.append(
                winner_weight / total_weight if total_weight > 0 else 0.0
            )

        consensus_confidence = (
            sum(position_agreements) / len(position_agreements)
            if position_agreements
            else 0.0
        )

        if consensus_confidence < self.consensus_threshold:
            return None

        track.emitted = True
        text = "".join(consensus_chars)
        last_reading = readings[-1]

        # Vehicle attribute consensus — confidence-weighted majority vote
        vehicle_color: str | None = None
        vehicle_type: str | None = None
        vehicle_make_model: str | None = None

        color_votes: Counter[str] = Counter()
        type_votes: Counter[str] = Counter()
        mm_votes: Counter[str] = Counter()

        for r in readings:
            if r.vehicle_color is not None:
                color_votes[r.vehicle_color] += r.vehicle_color_confidence
            if r.vehicle_type is not None:
                type_votes[r.vehicle_type] += r.vehicle_type_confidence
            if r.vehicle_make_model is not None:
                mm_votes[r.vehicle_make_model] += r.vehicle_make_model_confidence

        if color_votes:
            vehicle_color = color_votes.most_common(1)[0][0]
        if type_votes:
            vehicle_type = type_votes.most_common(1)[0][0]
        if mm_votes:
            vehicle_make_model = mm_votes.most_common(1)[0][0]

        return ConsensusResult(
            text=text,
            consensus_confidence=consensus_confidence,
            reading_count=len(readings),
            first_seen=readings[0].timestamp,
            last_seen=last_reading.timestamp,
            bbox=last_reading.bbox,
            source=last_reading.source,
            frame_number=last_reading.frame_number,
            vehicle_color=vehicle_color,
            vehicle_type=vehicle_type,
            vehicle_make_model=vehicle_make_model,
        )

    def prune_stale_tracks(self, now: float) -> None:
        """Remove tracks whose last reading is older than track_timeout."""
        self._tracks = [
            t
            for t in self._tracks
            if now - t.readings[-1].timestamp < self.track_timeout
        ]
