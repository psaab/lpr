"""Daemon mode for the LPR service."""

import logging
import os
import signal
import sys
from pathlib import Path

from .config import Config
from .pipeline import Pipeline

logger = logging.getLogger("lpr.daemon")


class Daemon:
    """Runs the LPR pipeline as a background daemon."""

    def __init__(self, config: Config):
        self.config = config
        self._running = False

    def start(self) -> None:
        """Start the daemon."""
        pid_file = Path(self.config.pid_file)

        # Check for existing PID file
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                # Check if process is still running
                os.kill(old_pid, 0)
                logger.error("Daemon already running with PID %d", old_pid)
                sys.exit(1)
            except (ProcessLookupError, ValueError):
                # Process not running, clean up stale PID file
                pid_file.unlink()

        # Write PID file
        pid_file.write_text(str(os.getpid()))
        logger.info("Daemon started with PID %d", os.getpid())

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._running = True

        try:
            pipeline = Pipeline(self.config)
            pipeline.run()
        finally:
            self._cleanup(pid_file)

    def _handle_signal(self, signum: int, _frame) -> None:
        """Handle termination signals."""
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down...", sig_name)
        self._running = False
        # Raise KeyboardInterrupt to stop the pipeline
        raise KeyboardInterrupt

    def _cleanup(self, pid_file: Path) -> None:
        """Clean up PID file on exit."""
        if pid_file.exists():
            pid_file.unlink()
        logger.info("Daemon stopped")
