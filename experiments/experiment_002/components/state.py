# File: components/state.py

"""State management for experiment"""

from dataclasses import dataclass, field
import torch
from typing import Dict, Any, Optional
from .logger import LoggerManager
from .base import BaseComponent, ExperimentError


@dataclass
class StateManager(BaseComponent):
    logger_manager: LoggerManager
    device: str = field(init=False)

    def __post_init__(self):
        """Initialize device and log the current state."""
        # Extract 'rank' and 'world_size' from config
        self.rank = self.config['distributed']['rank']
        self.world_size = self.config['distributed']['world_size']
        self.device = self._assign_device()
        self.log_state()

    def _assign_device(self) -> str:
        """Assign device based on rank and available GPUs."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                if self.logger_manager and self.is_main_process():
                    self.logger_manager.log_error("CUDA is available but no GPUs detected.")
                return "cpu"
            gpu_index = self.rank % num_gpus
            device_str = f"cuda:{gpu_index}"
            if self.logger_manager and self.is_main_process():
                self.logger_manager.log_info(f"Assigned device {device_str} to rank {self.rank}.")
            return device_str
        else:
            if self.logger_manager and self.is_main_process():
                self.logger_manager.log_info("CUDA not available. Using CPU.")
            return "cpu"

    def is_main_process(self) -> bool:
        """Check if this is the main process.
        
        Returns:
            bool: True if this is the main process (rank 0), False otherwise.
        """
        return self.rank == 0

    def log_state(self) -> None:
        """Log the current state of the experiment."""
        if self.logger_manager and self.is_main_process():
            self.logger_manager.log_info(
                f"StateManager initialized with Rank: {self.rank}, "
                f"World Size: {self.world_size}, Device: {self.device}"
            )