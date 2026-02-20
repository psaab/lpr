# LPR Project Planning

## Project Overview

LPR (License Plate Recognition) is a daemon that detects and reads license plates
from RTSP camera streams and video files using machine learning. It combines
YOLOv8 for plate detection with EasyOCR for text recognition, producing structured
JSON output suitable for downstream processing, logging, or alerting.

## Goals

- **Real-time plate detection** from live RTSP streams and recorded video files
- **GPU-accelerated inference** via CUDA when available, with automatic CPU fallback
- **Structured JSON output** for easy integration with other systems
- **Daemon mode** for long-running deployment with PID file and signal handling
- **CLI mode** for testing and one-off processing of video files
- **Modular design** with clean separation of concerns for maintainability

## Architecture Design

The project follows a modular pipeline architecture where each component has a
single responsibility:

```
src/lpr/
    __init__.py      # Package init, version
    cli.py           # Command-line interface (argparse)
    config.py        # Configuration management
    daemon.py        # Daemon mode with PID file and signal handling
    detector.py      # YOLOv8-based license plate detection
    output.py        # JSONL output writer (file, stdout)
    pipeline.py      # Orchestration, deduplication, tracking
    reader.py        # EasyOCR-based plate text recognition
    stream.py        # Video/RTSP frame capture using OpenCV
```

### Component Responsibilities

| Component     | Role                                                      |
|---------------|-----------------------------------------------------------|
| `stream.py`   | Opens video sources (RTSP URLs, video files), captures frames |
| `detector.py` | Runs YOLOv8 inference to locate plate bounding boxes      |
| `reader.py`   | Crops detected plates and runs EasyOCR for text extraction |
| `pipeline.py` | Connects components, deduplicates plates, manages tracking |
| `output.py`   | Writes detection results as JSONL to file or stdout       |
| `cli.py`      | Parses CLI arguments, dispatches to pipeline or daemon    |
| `daemon.py`   | Manages background execution, PID file, signal handling   |
| `config.py`   | Loads and validates configuration from CLI args and files  |

## Technology Choices

| Technology       | Version   | Purpose                              |
|------------------|-----------|--------------------------------------|
| Python           | 3.13+     | Runtime (3.10+ minimum)              |
| YOLOv8           | latest    | License plate detection (ultralytics)|
| EasyOCR          | latest    | Plate text recognition               |
| OpenCV (cv2)     | latest    | Video capture, image processing      |
| PyTorch          | latest    | ML backend (CPU and CUDA)            |

### Why These Choices

- **YOLOv8 (ultralytics)**: Best speed/accuracy tradeoff for object detection.
  Easy to use Python API, supports custom models, handles CUDA automatically.
- **EasyOCR**: Good accuracy for license plate text, supports GPU acceleration,
  simple API, handles multiple languages.
- **JSONL output format**: One JSON object per line. Easy to parse, stream, and
  append. Compatible with standard Unix tools (jq, grep).
- **OpenCV**: Mature library with native RTSP support, handles diverse video
  codecs and formats reliably.

## Implementation Phases

### Phase 1: Project Setup
- Directory structure and packaging (pyproject.toml)
- Dependency management
- Basic configuration framework

### Phase 2: Core Components
- `stream.py` -- frame capture from video files and RTSP streams
- `detector.py` -- YOLOv8 plate detection with confidence filtering
- `reader.py` -- EasyOCR text recognition on cropped plate images
- `output.py` -- JSONL result writer

### Phase 3: Pipeline Integration
- `pipeline.py` -- connect stream -> detect -> read -> output
- Plate deduplication and tracking across frames
- Confidence thresholds and filtering

### Phase 4: CLI and Daemon Mode
- `cli.py` -- argument parsing, mode selection
- `daemon.py` -- background execution with PID file
- Signal handling (SIGTERM, SIGINT for graceful shutdown)

### Phase 5: Testing and Documentation
- Unit tests for each component
- Integration tests with sample video
- Project documentation (README, architecture, planning)

## Future Enhancements

- Fine-tuned YOLOv8 model trained specifically on license plates
- Plate tracking across frames (reduce duplicate OCR calls)
- Database output (SQLite, PostgreSQL)
- Web dashboard for live monitoring
- Multi-camera support with source labeling
- Webhook/MQTT notifications on plate detection
