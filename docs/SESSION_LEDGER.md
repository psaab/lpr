# Session Ledger

## 2026-02-20 -- Accuracy Improvements (v0.2.0)

### Session Summary

Upgraded the LPR daemon from v0.1.0 to v0.2.0 with four major accuracy
improvements: dedicated plate detection, purpose-built plate OCR, multi-frame
consensus voting, and deskew preprocessing.

### Environment

- **OS**: Linux (Debian-based)
- **Python**: 3.13
- **GPU**: NVIDIA (CUDA via ONNX Runtime)

### Decisions Made

| Decision | Rationale |
|----------|-----------|
| open-image-models YOLOv9 for detection | 96.6% mAP50 on plate detection, direct plate bbox without vehicle detection step, ONNX Runtime backend |
| fast-plate-ocr MobileViT-v2 for OCR | Purpose-built for license plates, per-character confidence, ONNX Runtime backend, faster than EasyOCR |
| ONNX Runtime instead of PyTorch | Lighter dependency, both new models use ONNX, CUDAExecutionProvider for GPU |
| Character-level consensus voting | More robust than string-level voting -- individual OCR errors get outvoted across frames |
| Weighted majority vote by confidence | Higher-confidence readings have more influence on consensus |
| Deskew via minAreaRect | Simple and effective for moderate rotation, skip extreme angles to avoid false corrections |
| Legacy fallback flags | Backward compatibility for environments where new packages aren't installed |
| Keep format correction logic | Still valuable on top of fast-plate-ocr for US plate pattern enforcement |

### What Changed

- **pyproject.toml**: Replaced ultralytics/easyocr/torch with open-image-models/fast-plate-ocr/onnxruntime-gpu
- **config.py**: Added consensus, engine selection, and detection model fields; torch import now optional
- **cli.py**: Added --min-readings, --consensus-threshold, --ocr-engine, --detection-model, --legacy-detector
- **output.py**: Added consensus_count and consensus_confidence to DetectionRecord
- **consensus.py**: NEW -- PlateConsensus with IoU/centroid tracking and character-level majority vote
- **detector.py**: Primary path uses open-image-models LicensePlateDetector; legacy YOLO+contour preserved
- **reader.py**: Primary OCR via fast-plate-ocr with deskew preprocessing; EasyOCR preserved as fallback
- **pipeline.py**: Replaced _seen_plates dedup with consensus system; passes new config to detector/reader

### Lessons Learned

- open-image-models returns DetectionResult objects with .bounding_box.{x1,y1,x2,y2} API
- fast-plate-ocr pads output with '_' characters that must be stripped; confidence is per-character-slot
- Both packages use ONNX Runtime and auto-detect CUDA, no torch dependency needed
- Consensus voting eliminates the need for simple time-based deduplication
- Deskew with minAreaRect is effective but needs angle normalization and bounds checking

---

## 2026-02-19 -- Initial Build

### Session Summary

Built the LPR (License Plate Recognition) daemon from scratch in a single
session. The project went from an empty directory to a fully functional pipeline
with CLI, daemon mode, and comprehensive documentation.

### Environment

- **OS**: Linux (Debian-based)
- **Python**: 3.13
- **GPU**: None on dev machine (CPU inference for testing)
- **Target**: Systems with NVIDIA GPUs for production use

### Decisions Made

| Decision | Rationale |
|----------|-----------|
| YOLOv8 for plate detection | Best speed/accuracy tradeoff, easy Python API via ultralytics, supports custom fine-tuned models |
| EasyOCR for text recognition | Good accuracy on plate text, GPU support, simple setup, multi-language capable |
| JSONL output format | One JSON object per line -- easy to parse, stream, and process with standard tools (jq, grep) |
| OpenCV for video capture | Mature, handles RTSP natively, broad codec support, reliable |
| Daemon mode with PID file | Standard Unix daemon pattern, clean signal handling for long-running deployment |
| Python 3.13 (3.10+ minimum) | Modern Python features, broad compatibility with ML libraries |
| Modular pipeline design | Each component (stream, detect, read, output) is independent and testable |
| Threaded frame capture | Prevents pipeline from blocking on I/O, handles RTSP buffering |
| Auto device detection | Uses CUDA when available, falls back to CPU, overridable via CLI |

### Architecture

Modular pipeline design with separate concerns:

```
stream -> detector -> reader -> pipeline (dedup) -> output
```

Each component is a standalone module that can be tested and developed
independently. The pipeline orchestrates the flow and handles deduplication.

### What Was Built

- **Project structure**: `src/lpr/` package with pyproject.toml
- **stream.py**: Frame capture from RTSP streams and video files
- **detector.py**: YOLOv8-based license plate detection
- **reader.py**: EasyOCR-based plate text recognition
- **pipeline.py**: Pipeline orchestration with dedup and tracking
- **output.py**: JSONL output writer (file and stdout)
- **cli.py**: Command-line interface with argparse
- **daemon.py**: Daemon mode with PID file and signal handling
- **config.py**: Configuration management
- **tests/**: Unit test framework
- **docs/**: Planning, architecture, and session documentation

### Lessons Learned

- EasyOCR initialization is slow (loads models on first use). Initialize once
  and reuse the reader instance across frames.
- YOLOv8 ultralytics API handles device selection internally, but explicit
  device passing ensures consistency with EasyOCR.
- RTSP streams need a separate capture thread to avoid frame buffering issues
  that cause stale frames and increasing latency.
- Plate deduplication is essential for RTSP streams -- the same plate appears
  in many consecutive frames and would flood the output without filtering.

### Next Steps

- Fine-tune a YOLOv8 model on a license plate dataset for better detection
- Add plate tracking across frames to reduce redundant OCR calls
- Database output (SQLite for local, PostgreSQL for networked)
- Web dashboard for live monitoring and search
- Multi-camera support with source labeling
- Webhook/MQTT notifications on plate detection
- Plate format validation (regex patterns per country/state)
