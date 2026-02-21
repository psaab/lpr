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
    |    | Deskew  |   reader.py
    |    +---------+
    |         |
    |         v
    |    +----+----+
    |    | Reader  |   reader.py
    |    | (OCR)   |
    |    +---------+
    |         |
    |         v
    |    +----+------+
    |    | Consensus |  consensus.py
    |    +-----------+
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
Frame Capture (OpenCV)                 stream.py
    |
    v
Plate Detection (YOLOv9 / open-image-models)  detector.py
    |  returns: plate bounding boxes + confidence scores
    v
Plate Crop                             detector.py
    |  extracts plate region from frame
    v
Deskew Preprocessing                   reader.py
    |  straightens rotated plates via minAreaRect
    v
OCR Recognition (fast-plate-ocr)       reader.py
    |  returns: plate text + per-character confidence
    v
Format Correction                      reader.py
    |  fixes OCR confusables using known plate formats
    v
Consensus Voting                       consensus.py
    |  character-level majority vote across multiple frames
    |  emits once per plate track when agreement threshold met
    v
JSON Output                            output.py
    |  writes JSONL record to file
    v
Consumer (log, database, alert)
```

### Detection Record Format

Each detection produces a JSON object:

```json
{
  "plate_text": "93508B3",
  "confidence": 0.94,
  "timestamp": 1.5,
  "frame_number": 45,
  "bbox": [120, 340, 280, 390],
  "source": "rtsp://camera1.local/stream",
  "detected_at": "2026-02-20T14:30:05.123456+00:00",
  "consensus_count": 5,
  "consensus_confidence": 0.95
}
```

## ML Model Details

### Detection: YOLOv9 License Plate Detector

- **Package**: `open-image-models` (ONNX Runtime backend)
- **Model**: `yolo-v9-t-384-license-plate-end2end` (96.6% mAP50)
- **Task**: Direct license plate bounding box detection (no vehicle detection step)
- **Input**: Video frame (BGR, any resolution)
- **Output**: Plate bounding boxes with confidence scores
- **Runtime**: ONNX Runtime with CUDAExecutionProvider (auto-detected)
- **Legacy fallback**: YOLOv8n via ultralytics (vehicle detection + contour extraction)

### Recognition: fast-plate-ocr

- **Package**: `fast-plate-ocr` (ONNX Runtime backend)
- **Model**: `global-plates-mobile-vit-v2-model` (MobileViT-v2 architecture)
- **Task**: Reading text from cropped plate images
- **Input**: Preprocessed grayscale plate image
- **Output**: Plate text string + per-character confidence scores
- **Runtime**: ONNX Runtime with CUDAExecutionProvider (auto-detected)
- **Legacy fallback**: EasyOCR with multi-variant preprocessing

### Preprocessing Pipeline

1. **Deskew**: Detect plate rotation via `cv2.minAreaRect`, correct angles up to 15°
2. **Grayscale conversion** + upscaling (small plates → 200px minimum width)
3. **CLAHE enhancement** + bilateral denoising (primary variant)
4. **Adaptive threshold** and **Otsu threshold** (additional variants for EasyOCR)

### Consensus Voting

- Tracks plates across frames using IoU (≥0.3) and centroid distance (≤100px) matching
- Character-level weighted majority vote using per-reading confidence
- Filters readings to dominant text length before voting
- Configurable minimum readings (default: 3) and agreement threshold (default: 0.6)
- Single emission per plate track (no duplicates)

## Threading Model

```
Main Thread                     Capture Thread
+-----------+                   +--------------+
| Pipeline  |<--- frames ------|  Stream      |
| loop      |    (queue)       |  capture     |
|           |                   |  loop        |
| detect()  |                   |              |
| read()    |                   | cv2.read()   |
| consensus |                   |              |
| output()  |                   |              |
+-----------+                   +--------------+
```

- **Capture thread**: Continuously reads frames from the video source into a
  bounded queue. This prevents the pipeline from blocking on I/O and handles
  RTSP stream buffering.
- **Main thread**: Pulls frames from the queue, runs detection, OCR, consensus
  voting, and writes output. Runs the pipeline loop and handles signal-based
  shutdown.
- **Queue**: Bounded to prevent memory growth when processing is slower than
  capture. Oldest frames are dropped when the queue is full.

## Device Selection

The system auto-detects the best available compute device:

1. Check for CUDA via ONNX Runtime CUDAExecutionProvider (primary models)
2. Optionally check `torch.cuda.is_available()` (legacy models)
3. If CUDA is available, use GPU for inference
4. If no GPU, fall back to CPU inference
5. Device can be overridden via `--device cpu` or `--device cuda`

## Configuration Options Reference

| Option               | CLI Flag                | Default                                    | Description                          |
|----------------------|-------------------------|--------------------------------------------|--------------------------------------|
| Video source         | positional              | (required)                                 | RTSP URL or path to video file       |
| Output file          | `--output`              | `detections.jsonl`                         | Path to JSONL output file            |
| Confidence threshold | `--confidence`          | `0.5`                                      | Minimum detection confidence         |
| Device               | `--device`              | `auto`                                     | Compute device: auto, cpu, cuda      |
| Frame skip           | `--frame-skip`          | `5`                                        | Process every Nth frame              |
| Dedup window         | `--dedup-seconds`       | `5`                                        | Suppress duplicate plates for N secs |
| Min readings         | `--min-readings`        | `3`                                        | Readings before consensus vote       |
| Consensus threshold  | `--consensus-threshold` | `0.6`                                      | Minimum agreement for consensus      |
| OCR engine           | `--ocr-engine`          | `fast-plate-ocr`                           | OCR backend: fast-plate-ocr, easyocr |
| Detection model      | `--detection-model`     | `yolo-v9-t-384-license-plate-end2end`      | open-image-models model name         |
| Legacy detector      | `--legacy-detector`     | off                                        | Use YOLOv8 + contour extraction      |
| Daemon mode          | `--daemon`              | off                                        | Run as background daemon             |
| PID file             | `--pid-file`            | `/tmp/lpr.pid`                             | PID file path (daemon mode)          |
| Log level            | `--log-level`           | `INFO`                                     | Logging verbosity                    |

## Error Handling

- **Stream disconnection**: Automatic reconnection with exponential backoff
  (RTSP streams). Exit with error for video files.
- **Detection failure**: Log warning, skip frame, continue processing.
- **OCR failure**: Log warning, record detection without plate text.
- **Model fallback**: If open-image-models or fast-plate-ocr unavailable,
  automatically falls back to legacy ultralytics/easyocr backends.
- **Signal handling**: SIGTERM and SIGINT trigger graceful shutdown -- finish
  current frame, flush output, remove PID file, exit cleanly.
