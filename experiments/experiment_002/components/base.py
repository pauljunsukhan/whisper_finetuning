# File: components/base.py

"""Base components and utilities for the experiment."""

from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

class ExperimentError(Exception):
    """Base exception class for experiment errors."""
    pass

@dataclass
class BaseComponent:
    """Base class for experiment components."""
    config: Dict[str, Any]

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key (str): Configuration key in dot notation (e.g., "training.batch_size")
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value or default
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def ensure_dir(self, path: Path) -> Path:
        """Ensure a directory exists.
        
        Args:
            path (Path): Directory path to ensure
            
        Returns:
            Path: The ensured directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message (str): Message to log
        """
        if hasattr(self, 'logger_manager') and self.logger_manager:
            self.logger_manager.log_info(message)

    def log_error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message (str): Message to log
        """
        if hasattr(self, 'logger_manager') and self.logger_manager:
            self.logger_manager.log_error(message)
