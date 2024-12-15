"""
Experiment 002: Fine-tuning Whisper Large V3 with expanded dataset.

Version: 0.2.0
Author: Your Name <your.email@example.com>
License: MIT
Description: This experiment focuses on fine-tuning the Whisper Large V3 model using an expanded dataset with improved regularization techniques.

Components:
- TrainerManager: Manages the training loop and evaluation.
- ModelManager: Handles model loading, saving, and configuration.
- DataManager: Manages dataset preparation and preprocessing.
- LoggerManager: Handles logging across the experiment.
- StateManager: Manages experiment state, especially in distributed settings.
- ExperimentError: Custom exception class for experiment-related errors.
"""

from .components.trainer import TrainerManager
from .components.model import ModelManager
from .components.data import DataManager
from .components.logger import LoggerManager
from .components.state import StateManager
from .components.base import ExperimentError

__all__ = [
    "TrainerManager",
    "ModelManager",
    "DataManager",
    "LoggerManager",
    "StateManager",
    "ExperimentError",
]

__version__ = "0.2.0"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"
__description__ = "Fine-tuning Whisper Large V3 with expanded dataset and improved regularization."
