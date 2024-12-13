"""Base classes for experiment components"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

class ExperimentError(Exception):
    """Base error class for experiments"""
    pass

class BaseComponent:
    """Base class for experiment components"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize component
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ExperimentError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ExperimentError(f"Config must be a dictionary, got {type(config)}")
        self.config = config
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get value from config with default
        
        Args:
            key: Config key (e.g. "training.batch_size")
            default: Default value if not found
            
        Returns:
            Config value
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for key in keys:
                if not isinstance(value, dict):
                    return default
                value = value.get(key)
                if value is None:
                    return default
                    
            return value if value is not None else default
            
        except Exception:
            return default
    
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def log_error(self, error: str) -> None:
        """Log error message"""
        logging.error(f"{self.__class__.__name__}: {error}")
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        logging.info(f"{self.__class__.__name__}: {message}")
    
    def log_debug(self, message: str) -> None:
        """Log debug message"""
        logging.debug(f"{self.__class__.__name__}: {message}")
        
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(config={self.config})" 