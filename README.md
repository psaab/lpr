# LPR -- License Plate Recognition

A daemon that detects and reads license plates from RTSP camera streams and video
files using machine learning. Uses a dedicated YOLOv9 plate detector and
fast-plate-ocr for text recognition, with multi-frame consensus voting for
accuracy.

## Features

- Real-time license plate detection from RTSP streams and video files
- Dedicated YOLOv9 plate detector via open-image-models (96.6% mAP50)
- Purpose-built plate OCR via fast-plate-ocr (MobileViT-v2)
- Multi-frame consensus voting with character-level majority vote
- Deskew preprocessing to straighten rotated plates
- Automatic CUDA GPU acceleration via ONNX Runtime with CPU fallback
- JSONL structured output (one JSON record per detection)
- Format-aware character correction for known US plate patterns
- Daemon mode with PID file and graceful signal handling
- Configurable frame skipping for performance tuning
- Automatic RTSP stream reconnection with backoff
- Legacy fallback to YOLOv8 + EasyOCR when new packages unavailable

## Quick Start

### Install

```bash
# Clone and install
git clone <repo-url> lpr
cd lpr
pip install -e .

# Or install with legacy model support (YOLOv8 + EasyOCR)
pip install -e ".[legacy]"
```

### Run

```bash
# Process a video file
lpr /path/to/video.mp4

# Connect to an RTSP stream
lpr rtsp://camera.local/stream

# Save output to a file
lpr rtsp://camera.local/stream -o detections.jsonl

# Run as a daemon
lpr rtsp://camera.local/stream -o /var/log/lpr/detections.jsonl --daemon
```

## Usage Examples

### Process a Video File

```bash
lpr dashcam.mp4 -c 0.6
```

### Monitor an RTSP Camera

```bash
lpr rtsp://192.168.1.100:554/stream1 \
    -o plates.jsonl \
    --min-readings 5 \
    --frame-skip 2
```

### Use Legacy Detection (YOLOv8 + EasyOCR)

```bash
lpr video.mp4 --legacy-detector --ocr-engine easyocr
```

### Daemon Mode

```bash
# Start as daemon
lpr rtsp://192.168.1.100:554/stream1 \
    -o /var/log/lpr/plates.jsonl \
    --daemon \
    --pid-file /var/run/lpr.pid

# Stop the daemon
kill $(cat /var/run/lpr.pid)
```

### Force CPU Inference

```bash
lpr video.mp4 --device cpu
```

## Configuration Options

| Option               | Flag                    | Default                               | Description                              |
|----------------------|-------------------------|---------------------------------------|------------------------------------------|
| Video source         | positional              | (required)                            | RTSP URL or path to video file           |
| Output file          | `-o, --output`          | `detections.jsonl`                    | Path to JSONL output file                |
| Confidence threshold | `-c, --confidence`      | `0.5`                                 | Minimum detection confidence (0-1)       |
| Device               | `-d, --device`          | `auto`                                | Compute device: auto, cpu, cuda          |
| Frame skip           | `-s, --frame-skip`      | `5`                                   | Process every Nth frame                  |
| Min readings         | `--min-readings`        | `3`                                   | Readings required before consensus vote  |
| Consensus threshold  | `--consensus-threshold` | `0.6`                                 | Minimum character agreement ratio        |
| OCR engine           | `--ocr-engine`          | `fast-plate-ocr`                      | OCR backend: fast-plate-ocr or easyocr   |
| Detection model      | `--detection-model`     | `yolo-v9-t-384-license-plate-end2end` | open-image-models model name             |
| Legacy detector      | `--legacy-detector`     | off                                   | Use YOLOv8 + contour plate extraction    |
| Dedup window         | `--dedup-seconds`       | `5`                                   | Suppress duplicate plates for N secs     |
| Daemon mode          | `--daemon`              | off                                   | Run as background daemon                 |
| PID file             | `--pid-file`            | `/tmp/lpr.pid`                        | PID file path (daemon mode)              |
| Log level            | `--log-level`           | `INFO`                                | Logging: DEBUG, INFO, WARNING, ERROR     |

## Output Format

Each detection is written as a single JSON line (JSONL format):

```json
{"plate_text": "93508B3", "confidence": 0.94, "timestamp": 1.5, "frame_number": 45, "bbox": [120, 340, 280, 390], "source": "video.mp4", "detected_at": "2026-02-20T14:30:05+00:00", "consensus_count": 5, "consensus_confidence": 0.95}
```

Fields:

| Field                  | Type   | Description                              |
|------------------------|--------|------------------------------------------|
| `plate_text`           | string | Recognized plate text                    |
| `confidence`           | float  | Best individual OCR confidence (0-1)     |
| `timestamp`            | float  | Video timestamp in seconds               |
| `frame_number`         | int    | Frame index in the video stream          |
| `bbox`                 | array  | Bounding box [x1, y1, x2, y2] in pixels |
| `source`               | string | Video source URL or file path            |
| `detected_at`          | string | ISO 8601 wall-clock detection time       |
| `consensus_count`      | int    | Number of readings in consensus vote     |
| `consensus_confidence` | float  | Character-level agreement ratio (0-1)    |

Process output with standard tools:

```bash
# Watch detections live
tail -f plates.jsonl | jq .

# Find a specific plate
grep "93508B3" plates.jsonl | jq .

# Count detections per plate
jq -r .plate_text plates.jsonl | sort | uniq -c | sort -rn
```

## Requirements

- Python 3.10 or later
- ffmpeg (for video codec support)
- Optional: NVIDIA GPU with CUDA for accelerated inference

### Python Dependencies

- open-image-models (YOLOv9 plate detection, ONNX Runtime)
- fast-plate-ocr (MobileViT-v2 plate OCR, ONNX Runtime)
- onnxruntime-gpu (CUDA inference)
- opencv-python-headless
- numpy
- supervision

### Optional (Legacy)

- ultralytics (YOLOv8)
- easyocr
- torch / torchvision

## License

MIT
