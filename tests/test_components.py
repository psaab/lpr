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
    """Test that PlateReader preprocessing works."""
    from lpr.reader import PlateReader

    reader = PlateReader(device="cpu")

    # Create a small test plate image
    plate = np.ones((30, 100, 3), dtype=np.uint8) * 255
    cv2.putText(plate, "TEST", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    processed = reader._preprocess(plate)
    assert len(processed.shape) == 2  # grayscale
    assert processed.shape[1] >= 100  # at least 100px wide


def test_config_resolve_device():
    """Test device auto-detection."""
    from lpr.config import Config

    config = Config(device="cpu")
    assert config.resolve_device() == "cpu"

    config = Config(device="auto")
    device = config.resolve_device()
    assert device in ("cpu", "cuda")
