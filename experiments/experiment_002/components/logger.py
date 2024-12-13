"""Logging and metric tracking components"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
from transformers import TrainerCallback, TrainerState, TrainerControl

@dataclass
class LoggerManager:
    """Manages logging and metrics"""
    config: Dict[str, Any]
    rank: int = 0
    log_dir: Optional[Path] = None
    logger: logging.Logger = field(init=False)
    
    def __post_init__(self) -> None:
        try:
            # Construct full path
            output_dir = Path(self.config["logging"]["output_dir"])
            experiment_name = self.config["experiment"]["name"]
            log_dir = Path(self.config["logging"]["log_dir"])
            
            # Combine paths and ensure it's a Path object
            self.log_dir = Path(output_dir / experiment_name / log_dir)
            
            # Only rank 0 creates the directory
            if self.rank == 0 and self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger
            self.logger = logging.getLogger(f"experiment_rank_{self.rank}")
            self.logger.setLevel(logging.INFO)
            
            # Remove any existing handlers
            for handler in self.logger.handlers[:]:
                handler.close()  # Properly close handlers
                self.logger.removeHandler(handler)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Create and add file handler only for rank 0
            if self.rank == 0:
                try:
                    fh = logging.FileHandler(self.log_dir / "experiment.log")
                    fh.setLevel(logging.INFO)
                    fh.setFormatter(formatter)
                    self.logger.addHandler(fh)
                except (OSError, PermissionError) as e:
                    print(f"Warning: Could not create log file: {e}")
                    print("Continuing with console logging only")
            
            # Create and add stream handler for all ranks
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
            
            # Prevent propagation to root logger
            self.logger.propagate = False
            
        except Exception as e:
            print(f"Error initializing logger: {e}")
            # Fallback to basic console logging
            self.logger = logging.getLogger(f"experiment_rank_{self.rank}")
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
    
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
        if self.rank == 0:
            self.logger.info(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        if self.rank == 0:
            self.logger.error(message)
    
    def log_metric(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics"""
        if self.rank == 0:
            metric_file = self.log_dir / self.config["logging"]["paths"]["metric_file"]
            entry = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "metrics": metrics
            }
            with metric_file.open("a") as f:
                json.dump(entry, f)
                f.write("\n")
    
    @classmethod
    def load_temp_logger(cls) -> 'LoggerManager':
        """Initialize a temporary logger for early stages."""
        temp_logger = logging.getLogger("temp_logger")
        temp_logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Create formatter and add to handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        if not temp_logger.handlers:
            temp_logger.addHandler(ch)
        
        # Initialize temporary LoggerManager with a complete dummy config
        temp_logger_manager = cls(config={
            "logging": {
                "output_dir": "./outputs",
                "log_dir": "logs",
                "paths": {
                    "metric_file": "metrics.jsonl"
                }
            },
            "experiment": {
                "name": "temp"
            }
        })
        temp_logger_manager.logger = temp_logger  # Override the logger
        return temp_logger_manager

class InputShapeLoggerCallback(TrainerCallback):
    """TrainerCallback to log input tensor shapes during training."""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger_manager = logger_manager
        
    def on_batch_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ) -> None:
        inputs = kwargs.get('inputs', {})
        if 'input_features' in inputs:
            self.logger_manager.log_info(f"Training batch input_features shape: {inputs['input_features'].shape}")
        if 'decoder_input_ids' in inputs:
            self.logger_manager.log_info(f"Training batch decoder_input_ids shape: {inputs['decoder_input_ids'].shape}")
        if 'labels' in inputs:
            self.logger_manager.log_info(f"Training batch labels shape: {inputs['labels'].shape}")