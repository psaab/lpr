# LPR Architecture

## Component Diagram

```
+------------------+
|   CLI / Daemon   |    cli.py, daemon.py
|   (Entry Point)  |
+--------+---------+
         |
         v
+--------+---------+
|     Pipeline     |    pipeline.py
|  (Orchestrator)  |
+--------+---------+
         |
    +----+----+------------------+
    |         |                  |
    v         v                  v
+---+----+ +--+-------+  +------+----+
| Stream | | Detector |  |  Output   |   output.py
|        | |          |  | (JSONL)   |
+---+----+ +--+-------+  +-----------+
    |         |
    |         v
    |    +----+----+
    |    | Reader  |   reader.py
    |    | (OCR)   |
    |    +---------+
    |
    v
+---+------------+
| Video Source   |
| (RTSP / File)  |
+----------------+
```

## Data Flow

```
Video Source (RTSP/file)
    |
    v
Frame Capture (OpenCV)          stream.py
    |
    v
Plate Detection (YOLOv8)        detector.py
    |  returns: bounding boxes + confidence scores
    v
Plate Crop (OpenCV)              pipeline.py
    |  extracts plate region from frame
    v
OCR Recognition (EasyOCR)        reader.py
    |  returns: plate text + confidence
    v
Deduplication / Tracking         pipeline.py
    |  filters duplicate reads within time window
    v
JSON Output                      output.py
    |  writes JSONL record to file or stdout
    v
Consumer (log, database, alert)
```

### Detection Record Format

Each detection produces a JSON object:

```json
{
  "timestamp": "2026-02-19T14:30:05.123456",
  "plate_text": "ABC1234",
  "confidence": 0.94,
  "bbox": [120, 340, 280, 390],
  "source": "rtsp://camera1.local/stream",
  "frame_number": 4521
}
```

## ML Model Details

### Detection: YOLOv8

- **Model**: YOLOv8n (nano) or YOLOv8s (small) from ultralytics
- **Task**: Object detection -- locating license plates in video frames
- **Input**: Video frame (BGR, any resolution)
- **Output**: Bounding boxes with confidence scores
- **Performance**: YOLOv8n runs at 30+ FPS on modern GPUs, 5-10 FPS on CPU
- **Custom models**: Supports loading fine-tuned `.pt` weights via `--model` flag

### Recognition: EasyOCR

- **Task**: Reading text from cropped plate images
- **Input**: Cropped plate image (BGR)
- **Output**: Recognized text string with confidence score
- **Languages**: Configurable, defaults to English (`en`)
- **GPU support**: Uses CUDA when available for faster inference

## Threading Model

```
Main Thread                     Capture Thread
+-----------+                   +--------------+
| Pipeline  |<--- frames ------|  Stream      |
| loop      |    (queue)       |  capture     |
|           |                   |  loop        |
| detect()  |                   |              |
| read()    |                   | cv2.read()   |
| output()  |                   |              |
+-----------+                   +--------------+
```

- **Capture thread**: Continuously reads frames from the video source into a
  bounded queue. This prevents the pipeline from blocking on I/O and handles
  RTSP stream buffering.
- **Main thread**: Pulls frames from the queue, runs detection and OCR, writes
  output. Runs the pipeline loop and handles signal-based shutdown.
- **Queue**: Bounded to prevent memory growth when processing is slower than
  capture. Oldest frames are dropped when the queue is full.

## Device Selection

The system auto-detects the best available compute device:

1. Check for CUDA-capable GPU via `torch.cuda.is_available()`
2. If CUDA is available, use GPU for both YOLOv8 and EasyOCR
3. If no GPU, fall back to CPU inference
4. Device can be overridden via `--device cpu` or `--device cuda`

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Configuration Options Reference

| Option              | CLI Flag            | Default        | Description                          |
|---------------------|---------------------|----------------|--------------------------------------|
| Video source        | `--source`          | (required)     | RTSP URL or path to video file       |
| Output file         | `--output`          | stdout         | Path to JSONL output file            |
| Detection model     | `--model`           | `yolov8n.pt`   | Path to YOLOv8 weights               |
| Confidence threshold| `--confidence`      | `0.5`          | Minimum detection confidence         |
| Device              | `--device`          | `auto`         | Compute device: auto, cpu, cuda      |
| Frame skip          | `--frame-skip`      | `0`            | Process every Nth frame (0 = all)    |
| Dedup window        | `--dedup-seconds`   | `5`            | Suppress duplicate plates for N secs |
| Daemon mode         | `--daemon`          | `false`        | Run as background daemon             |
| PID file            | `--pid-file`        | `/tmp/lpr.pid` | PID file path (daemon mode)          |
| Log level           | `--log-level`       | `INFO`         | Logging verbosity                    |
| OCR languages       | `--languages`       | `en`           | Comma-separated EasyOCR languages    |

## Error Handling

- **Stream disconnection**: Automatic reconnection with exponential backoff
  (RTSP streams). Exit with error for video files.
- **Detection failure**: Log warning, skip frame, continue processing.
- **OCR failure**: Log warning, record detection without plate text.
- **Signal handling**: SIGTERM and SIGINT trigger graceful shutdown -- finish
  current frame, flush output, remove PID file, exit cleanly.
