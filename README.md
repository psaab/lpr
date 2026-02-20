# LPR -- License Plate Recognition

A daemon that detects and reads license plates from RTSP camera streams and video
files using machine learning. Uses YOLOv8 for plate detection and EasyOCR for text
recognition.

## Features

- Real-time license plate detection from RTSP streams and video files
- YOLOv8 object detection with configurable model weights
- EasyOCR text recognition with multi-language support
- Automatic CUDA GPU acceleration with CPU fallback
- JSONL structured output (one JSON record per detection)
- Plate deduplication to suppress repeated reads within a time window
- Daemon mode with PID file and graceful signal handling
- Configurable frame skipping for performance tuning
- Automatic RTSP stream reconnection with backoff

## Quick Start

### Install

```bash
# Clone and install
git clone <repo-url> lpr
cd lpr
pip install -e .

# Or install with CUDA support (requires NVIDIA GPU + drivers)
pip install -e ".[cuda]"
```

### Run

```bash
# Process a video file
lpr --source /path/to/video.mp4

# Connect to an RTSP stream
lpr --source rtsp://camera.local/stream

# Save output to a file
lpr --source rtsp://camera.local/stream --output detections.jsonl

# Run as a daemon
lpr --source rtsp://camera.local/stream --output /var/log/lpr/detections.jsonl --daemon
```

## Usage Examples

### Process a Video File

```bash
lpr --source dashcam.mp4 --confidence 0.6
```

### Monitor an RTSP Camera

```bash
lpr --source rtsp://192.168.1.100:554/stream1 \
    --output plates.jsonl \
    --dedup-seconds 10 \
    --frame-skip 2
```

### Daemon Mode

```bash
# Start as daemon
lpr --source rtsp://192.168.1.100:554/stream1 \
    --output /var/log/lpr/plates.jsonl \
    --daemon \
    --pid-file /var/run/lpr.pid

# Stop the daemon
kill $(cat /var/run/lpr.pid)
```

### Use a Custom Detection Model

```bash
lpr --source video.mp4 --model /path/to/custom-plates.pt
```

### Force CPU Inference

```bash
lpr --source video.mp4 --device cpu
```

## Configuration Options

| Option              | Flag                | Default        | Description                          |
|---------------------|---------------------|----------------|--------------------------------------|
| Video source        | `--source`          | (required)     | RTSP URL or path to video file       |
| Output file         | `--output`          | stdout         | Path to JSONL output file            |
| Detection model     | `--model`           | `yolov8n.pt`   | Path to YOLOv8 weights               |
| Confidence threshold| `--confidence`      | `0.5`          | Minimum detection confidence (0-1)   |
| Device              | `--device`          | `auto`         | Compute device: auto, cpu, cuda      |
| Frame skip          | `--frame-skip`      | `0`            | Process every Nth frame (0 = all)    |
| Dedup window        | `--dedup-seconds`   | `5`            | Suppress duplicate plates for N secs |
| Daemon mode         | `--daemon`          | off            | Run as background daemon             |
| PID file            | `--pid-file`        | `/tmp/lpr.pid` | PID file path (daemon mode)          |
| Log level           | `--log-level`       | `INFO`         | Logging: DEBUG, INFO, WARNING, ERROR |
| OCR languages       | `--languages`       | `en`           | Comma-separated EasyOCR language codes|

## Output Format

Each detection is written as a single JSON line (JSONL format):

```json
{"timestamp": "2026-02-19T14:30:05.123456", "plate_text": "ABC1234", "confidence": 0.94, "bbox": [120, 340, 280, 390], "source": "rtsp://camera.local/stream", "frame_number": 4521}
```

Fields:

| Field          | Type   | Description                              |
|----------------|--------|------------------------------------------|
| `timestamp`    | string | ISO 8601 detection timestamp             |
| `plate_text`   | string | Recognized plate text                    |
| `confidence`   | float  | Combined detection + OCR confidence (0-1)|
| `bbox`         | array  | Bounding box [x1, y1, x2, y2] in pixels |
| `source`       | string | Video source URL or file path            |
| `frame_number` | int    | Frame index in the video stream          |

Process output with standard tools:

```bash
# Watch detections live
tail -f plates.jsonl | jq .

# Find a specific plate
grep "ABC1234" plates.jsonl | jq .

# Count detections per plate
jq -r .plate_text plates.jsonl | sort | uniq -c | sort -rn
```

## Requirements

- Python 3.10 or later
- ffmpeg (for video codec support)
- Optional: NVIDIA GPU with CUDA for accelerated inference

### Python Dependencies

- ultralytics (YOLOv8)
- easyocr
- opencv-python
- torch / torchvision

## License

MIT
