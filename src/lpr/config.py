"""Configuration for the LPR daemon."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    """LPR daemon configuration."""

    source: str = ""
    output_path: Path = field(default_factory=lambda: Path("detections.jsonl"))
    device: str = "auto"
    confidence_threshold: float = 0.5
    frame_skip: int = 5
    daemon: bool = False
    pid_file: Path = field(default_factory=lambda: Path("/tmp/lpr.pid"))
    log_level: str = "INFO"
    log_file: Path | None = None
    reconnect_delay: float = 5.0
    max_reconnects: int = 10
    dedup_seconds: float = 5.0

    def resolve_device(self) -> str:
        """Resolve device string, falling back to cpu when CUDA unavailable."""
        requested = self.device
        log = logging.getLogger("lpr")

        if requested in ("auto", "cuda"):
            if torch.cuda.is_available():
                log.info(
                    "CUDA available - using GPU: %s",
                    torch.cuda.get_device_name(0),
                )
                return "0"
            if requested == "cuda":
                log.warning(
                    "CUDA requested but not available - falling back to CPU"
                )
            else:
                log.info("CUDA not available - using CPU inference")
            return "cpu"
        return requested
