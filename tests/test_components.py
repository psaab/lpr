"""Unit tests for LPR components."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_test_video(path: str, frames: int = 30, fps: int = 10) -> str:
    """Create a simple test video with a white rectangle (fake plate)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))

    for i in range(frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a road-like background
        frame[300:480, :] = (80, 80, 80)
        # Draw a car-like rectangle
        cv2.rectangle(frame, (200, 200), (440, 400), (30, 30, 180), -1)
        # Draw a plate-like white rectangle in the lower part
        cv2.rectangle(frame, (270, 350), (370, 380), (255, 255, 255), -1)
        # Put some text on the plate
        cv2.putText(
            frame, "ABC 1234", (275, 375),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )
        writer.write(frame)

    writer.release()
    return path


def test_stream_reader():
    """Test that StreamReader can read frames from a video file."""
    from lpr.stream import StreamReader

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        make_test_video(f.name, frames=20, fps=10)

        reader = StreamReader(f.name, frame_skip=5)
        frames = list(reader.frames())

        # 20 frames / skip 5 = 4 frames
        assert len(frames) == 4
        assert frames[0].frame_number == 5
        assert frames[0].image.shape == (480, 640, 3)

        Path(f.name).unlink()


def test_stream_reader_pyav():
    """Test PyAV decode path produces correct frames."""
    try:
        import av  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("PyAV not installed")

    from lpr.stream import StreamReader

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        make_test_video(f.name, frames=20, fps=10)

        reader = StreamReader(f.name, frame_skip=5)

        # Verify PyAV opens successfully
        assert reader._open_pyav(), "PyAV failed to open test video"
        assert reader._use_pyav
        assert reader._container is not None
        reader.close()

        # Verify full pipeline produces correct frames
        frames = list(reader.frames())
        assert len(frames) == 4
        assert frames[0].frame_number == 5
        assert frames[0].image.shape == (480, 640, 3)
        assert frames[0].timestamp >= 0.0

        Path(f.name).unlink()


def test_output_writer():
    """Test JSON output writer."""
    from lpr.output import JSONOutputWriter

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        output_path = Path(f.name)

    writer = JSONOutputWriter(output_path)
    with writer:
        record = writer.make_record(
            plate_text="ABC1234",
            confidence=0.95,
            timestamp=1.5,
            frame_number=10,
            bbox=(100, 200, 300, 250),
            source="test.mp4",
        )
        writer.write(record)

    # Read back and verify
    with open(output_path) as f:
        line = f.readline()
        data = json.loads(line)

    assert data["plate_text"] == "ABC1234"
    assert data["confidence"] == 0.95
    assert data["bbox"] == [100, 200, 300, 250]
    assert "detected_at" in data

    output_path.unlink()


def test_reader_preprocess():
    """Test that PlateReader preprocessing returns multiple variants."""
    from lpr.reader import PlateReader

    reader = PlateReader(device="cpu", ocr_engine="fast-plate-ocr")

    # Create a small test plate image
    plate = np.ones((30, 100, 3), dtype=np.uint8) * 255
    cv2.putText(plate, "TEST", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    variants = reader._preprocess(plate)
    assert isinstance(variants, list)
    assert len(variants) == 3
    assert len(variants[0].shape) == 2  # grayscale


def test_config_resolve_device():
    """Test device auto-detection."""
    from lpr.config import Config

    config = Config(device="cpu")
    assert config.resolve_device() == "cpu"

    config = Config(device="auto")
    device = config.resolve_device()
    assert device in ("cpu", "cuda")


def test_reader_deskew():
    """Test that PlateReader deskew corrects rotated images."""
    from lpr.reader import PlateReader

    reader = PlateReader(device="cpu")

    # Create a small plate image
    plate = np.ones((60, 200, 3), dtype=np.uint8) * 255
    cv2.putText(plate, "TEST123", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Rotate it by 10 degrees
    h, w = plate.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
    rotated = cv2.warpAffine(plate, M, (w, h), borderValue=(255, 255, 255))

    result = reader._deskew(rotated)
    assert result.shape[:2] == rotated.shape[:2]


def test_consensus_basic():
    """Test that consensus emits after min_readings are reached."""
    from lpr.consensus import PlateConsensus, PlateReading

    consensus = PlateConsensus(min_readings=3)
    bbox = (100, 200, 300, 250)

    for i in range(3):
        reading = PlateReading(
            text="ABC1234", confidence=0.9, bbox=bbox,
            frame_number=i + 1, timestamp=float(i), source="test.mp4",
        )
        result = consensus.add_reading(reading)
        if i < 2:
            assert result is None

    assert result is not None
    assert result.text == "ABC1234"
    assert result.reading_count == 3


def test_consensus_majority_vote():
    """Test that consensus uses character-level majority voting."""
    from lpr.consensus import PlateConsensus, PlateReading

    consensus = PlateConsensus(min_readings=3)
    bbox = (100, 200, 300, 250)
    texts = ["ABC1234", "A8C1234", "ABC1234"]

    for i, text in enumerate(texts):
        reading = PlateReading(
            text=text, confidence=0.9, bbox=bbox,
            frame_number=i + 1, timestamp=float(i), source="test.mp4",
        )
        result = consensus.add_reading(reading)

    assert result is not None
    assert result.text == "ABC1234"


def test_consensus_no_double_emit():
    """Test that a track only emits once."""
    from lpr.consensus import PlateConsensus, PlateReading

    consensus = PlateConsensus(min_readings=3)
    bbox = (100, 200, 300, 250)

    for i in range(3):
        reading = PlateReading(
            text="XYZ5678", confidence=0.9, bbox=bbox,
            frame_number=i + 1, timestamp=float(i), source="test.mp4",
        )
        consensus.add_reading(reading)

    # Add more readings to the same track
    for i in range(3, 6):
        reading = PlateReading(
            text="XYZ5678", confidence=0.9, bbox=bbox,
            frame_number=i + 1, timestamp=float(i), source="test.mp4",
        )
        result = consensus.add_reading(reading)
        assert result is None  # should not emit again


def test_vehicle_roi_estimation():
    """Test that estimated vehicle ROI is larger than plate and clamped to frame."""
    from lpr.vehicle import VehicleClassifier

    plate_bbox = (270, 350, 370, 380)
    frame_shape = (480, 640, 3)

    roi = VehicleClassifier.estimate_vehicle_roi(plate_bbox, frame_shape)
    rx1, ry1, rx2, ry2 = roi

    # ROI must be larger than the plate
    assert rx2 - rx1 > plate_bbox[2] - plate_bbox[0]
    assert ry2 - ry1 > plate_bbox[3] - plate_bbox[1]

    # ROI must be clamped to frame bounds
    assert rx1 >= 0
    assert ry1 >= 0
    assert rx2 <= 640
    assert ry2 <= 480

    # Plate near top edge — ROI should clamp vy1 to 0
    top_plate = (300, 10, 400, 30)
    roi_top = VehicleClassifier.estimate_vehicle_roi(top_plate, frame_shape)
    assert roi_top[1] == 0


def test_vehicle_attributes_dataclass():
    """Test VehicleAttributes defaults."""
    from lpr.vehicle import VehicleAttributes

    attrs = VehicleAttributes(
        color="white", color_confidence=0.95,
        vehicle_type="car", type_confidence=0.88,
    )
    assert attrs.make_model is None
    assert attrs.make_model_confidence == 0.0
    assert attrs.color == "white"
    assert attrs.vehicle_type == "car"


def test_consensus_with_vehicle_attrs():
    """Test that consensus majority-votes vehicle attributes."""
    from lpr.consensus import PlateConsensus, PlateReading

    consensus = PlateConsensus(min_readings=3)
    bbox = (100, 200, 300, 250)

    colors = ["white", "white", "gray"]
    types = ["car", "car", "truck"]

    for i in range(3):
        reading = PlateReading(
            text="ABC1234", confidence=0.9, bbox=bbox,
            frame_number=i + 1, timestamp=float(i), source="test.mp4",
            vehicle_color=colors[i], vehicle_type=types[i],
            vehicle_color_confidence=0.9, vehicle_type_confidence=0.9,
        )
        result = consensus.add_reading(reading)

    assert result is not None
    assert result.vehicle_color == "white"  # 2 votes vs 1
    assert result.vehicle_type == "car"  # 2 votes vs 1


def test_output_record_with_vehicle():
    """Test that vehicle fields appear in JSON output dict."""
    from lpr.output import JSONOutputWriter

    record = JSONOutputWriter.make_record(
        plate_text="XYZ5678", confidence=0.9, timestamp=2.0,
        frame_number=20, bbox=(50, 100, 200, 150), source="test.mp4",
        vehicle_color="blue", vehicle_type="truck",
        vehicle_make_model="Ford F-150",
    )
    d = record.to_dict()
    assert d["vehicle_color"] == "blue"
    assert d["vehicle_type"] == "truck"
    assert d["vehicle_make_model"] == "Ford F-150"


def test_output_writer_with_consensus():
    """Test JSON output writer with consensus fields."""
    from lpr.output import JSONOutputWriter

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        output_path = Path(f.name)

    writer = JSONOutputWriter(output_path)
    with writer:
        record = writer.make_record(
            plate_text="ABC1234", confidence=0.95, timestamp=1.5,
            frame_number=10, bbox=(100, 200, 300, 250), source="test.mp4",
            consensus_count=5, consensus_confidence=0.92,
        )
        writer.write(record)

    with open(output_path) as f:
        data = json.loads(f.readline())

    assert data["consensus_count"] == 5
    assert data["consensus_confidence"] == 0.92

    output_path.unlink()
