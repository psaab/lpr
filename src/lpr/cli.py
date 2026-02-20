"""Command-line interface for the LPR daemon."""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config


def setup_logging(level: str, log_file: Path | None = None) -> None:
    """Configure logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse command-line arguments into a Config."""
    parser = argparse.ArgumentParser(
        prog="lpr",
        description="License Plate Recognition daemon",
    )
    parser.add_argument(
        "source",
        help="Video file path or RTSP stream URL",
    )
    parser.add_argument(
        "-o", "--output",
        default="detections.jsonl",
        help="Output JSONL file path (default: detections.jsonl)",
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device (default: auto-detect)",
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "-s", "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a background daemon",
    )
    parser.add_argument(
        "--pid-file",
        default="/tmp/lpr.pid",
        help="PID file path for daemon mode (default: /tmp/lpr.pid)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: stderr only)",
    )
    parser.add_argument(
        "--dedup-seconds",
        type=float,
        default=5.0,
        help="Seconds to suppress duplicate plate readings (default: 5.0)",
    )

    args = parser.parse_args(argv)

    return Config(
        source=args.source,
        output_path=Path(args.output),
        device=args.device,
        confidence_threshold=args.confidence,
        frame_skip=args.frame_skip,
        daemon=args.daemon,
        pid_file=Path(args.pid_file),
        log_level=args.log_level,
        log_file=Path(args.log_file) if args.log_file else None,
        dedup_seconds=args.dedup_seconds,
    )


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    config = parse_args(argv)
    setup_logging(config.log_level, config.log_file)

    logger = logging.getLogger("lpr")
    logger.info("LPR daemon starting")
    logger.info("Source: %s", config.source)
    logger.info("Output: %s", config.output_path)
    logger.info("Device: %s", config.device)

    if config.daemon:
        from .daemon import Daemon
        daemon = Daemon(config)
        daemon.start()
    else:
        from .pipeline import Pipeline
        pipeline = Pipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
