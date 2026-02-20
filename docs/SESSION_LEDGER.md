# Session Ledger

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
