# File: components/logger.py

"""Logging and metric tracking components"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseComponent, ExperimentError

@dataclass
class LoggerManager(BaseComponent):
    """Manages logging and metrics"""
    rank: int = 0
    log_dir: Optional[Path] = None
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        """Initialize logger with configuration."""
        try:
            # Retrieve log levels from config with defaults
            console_min_level = self.get_config("logging.console.min_level", "INFO").upper()
            file_min_level = self.get_config("logging.file.min_level", "INFO").upper()

            # Validate log levels
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if console_min_level not in valid_levels:
                raise ValueError(f"Invalid console log level: {console_min_level}")
            if file_min_level not in valid_levels:
                raise ValueError(f"Invalid file log level: {file_min_level}")

            # Construct full path
            output_dir = Path(self.get_config("logging.output_dir", "./outputs"))
            experiment_name = self.get_config("experiment.name", "experiment")
            log_dir_name = self.get_config("logging.log_dir", "logs")
            log_dir = Path(log_dir_name)

            # Combine paths and ensure it's a Path object
            self.log_dir = self.ensure_dir(output_dir / experiment_name / log_dir)

            # Create logger
            self.logger = logging.getLogger(f"experiment_rank_{self.rank}")
            self.logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter

            # Remove any existing handlers
            for handler in self.logger.handlers[:]:
                handler.close()  # Properly close handlers
                self.logger.removeHandler(handler)

            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Create and add file handler only for rank 0 if enabled
            if self.rank == 0 and self.get_config("logging.file.enabled", True):
                log_file_name = self.get_config("logging.paths.log_file", "experiment.log")
                try:
                    fh = logging.FileHandler(self.log_dir / log_file_name)
                    fh.setLevel(getattr(logging, file_min_level, logging.INFO))
                    fh.setFormatter(formatter)
                    self.logger.addHandler(fh)
                except (OSError, PermissionError) as e:
                    print(f"Warning: Could not create log file '{log_file_name}': {e}")
                    print("Continuing with console logging only")

            # Create and add stream handler for all ranks if enabled
            if self.get_config("logging.console.enabled", True):
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(getattr(logging, console_min_level, logging.INFO))
                sh.setFormatter(formatter)
                self.logger.addHandler(sh)

            # Prevent propagation to root logger
            self.logger.propagate = False

            # Log successful initialization
            self.log_info("LoggerManager initialized successfully.")

        except Exception as e:
            print(f"Error initializing LoggerManager: {e}")
            # Fallback to basic console logging
            self.logger = logging.getLogger(f"experiment_rank_{self.rank}_fallback")
            self.logger.setLevel(logging.INFO)
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
            self.logger.propagate = False
            self.logger.error(f"Failed to initialize LoggerManager: {e}")

    @classmethod
    def load_temp_logger(cls) -> 'LoggerManager':
        """Load a temporary logger for early-stage logging."""
        temp_logger = cls(config={}, rank=0)
        temp_logger.logger = logging.getLogger("temp_logger")
        temp_logger.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)
        temp_logger.logger.addHandler(sh)
        temp_logger.logger.propagate = False
        temp_logger.log_info("Temporary logger initialized.")
        return temp_logger

    def __del__(self):
        """Ensure proper cleanup of handlers"""
        if hasattr(self, 'logger'):
            for handler in self.logger.handlers[:]:
                try:
                    handler.close()
                    self.logger.removeHandler(handler)
                except Exception:
                    pass

    def log_info(self, message: str) -> None:
        """Log info message"""
        if self.rank == 0 and self.get_config("logging.console.enabled", True):
            self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log error message"""
        if self.rank == 0 and self.get_config("logging.console.enabled", True):
            self.logger.error(message)

    def log_metric(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics"""
        if self.rank == 0 and self.get_config("logging.metrics.enabled", True):
            metric_file = self.log_dir / self.get_config("logging.paths.metric_file", "metrics.jsonl")
            try:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "metrics": metrics
                }
                with metric_file.open("a") as f:
                    json.dump(entry, f)
                    f.write("\n")
            except Exception as e:
                self.log_error(f"Failed to log metrics: {e}")
