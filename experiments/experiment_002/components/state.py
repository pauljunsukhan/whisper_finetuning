"""State management for experiment"""

from dataclasses import dataclass
import torch
from typing import Optional
from .logger import LoggerManager

@dataclass
class StateManager:
    rank: int
    world_size: int
    logger_manager: Optional[LoggerManager] = None

    def is_distributed(self) -> bool:
        """Check if running in distributed mode"""
        return self.world_size > 1

    @property
    def device(self) -> str:
        """Get device for current process"""
        if torch.cuda.is_available():
            return f"cuda:{self.rank}"
        return "cpu"

    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0

    def log_state(self) -> None:
        """Log current state"""
        if self.logger_manager and self.is_main_process():
            self.logger_manager.log_info(
                f"StateManager initialized with Rank: {self.rank}, "
                f"World Size: {self.world_size}, Device: {self.device}"
            )