# File: components/__init__.py

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

from .trainer import TrainerManager
from .model import ModelManager
from .data import DataManager
from .logger import LoggerManager
from .state import StateManager
from .base import ExperimentError
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import torch
import evaluate
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    WhisperTokenizer
)
from datasets import Dataset, load_dataset, DatasetDict

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

@dataclass(frozen=False)
class ComponentContext:
    """Context for component initialization"""
    config: Dict[str, Any]
    rank: int = field(default=0)
    world_size: int = field(default=1)
    state_manager: Optional[StateManager] = field(default=None)
    logger_manager: Optional[LoggerManager] = field(default=None)
    model_manager: Optional[ModelManager] = field(default=None)
    data_manager: Optional[DataManager] = field(default=None)
    trainer: Optional[TrainerManager] = field(default=None)
    processor: Optional[WhisperProcessor] = field(default=None)

def create_compute_metrics(processor: WhisperProcessor):
    """Create compute_metrics function with access to processor."""
    tokenizer = processor.tokenizer
    def compute_metrics(pred: Any) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and references using batch_decode directly
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metrics

@contextmanager
def component_error_handler(component_name: str, context: ComponentContext):
    """Context manager to handle errors in component initialization."""
    try:
        yield
    except Exception as e:
        if context.logger_manager:
            context.logger_manager.log_error(f"{component_name} initialization failed: {str(e)}")
        raise

def initialize_logger_manager(context: ComponentContext) -> LoggerManager:
    """Initialize LoggerManager component."""
    return LoggerManager(
        config=context.config,
        rank=context.rank
    )

def initialize_state_manager(context: ComponentContext) -> StateManager:
    """Initialize StateManager component."""
    with component_error_handler("StateManager", context):
        if context.logger_manager is None:
            raise ExperimentError("LoggerManager must be initialized before StateManager.")
        return StateManager(
            config=context.config,
            logger_manager=context.logger_manager
        )

def initialize_model_manager(context: ComponentContext) -> ModelManager:
    """Initialize ModelManager component."""
    with component_error_handler("ModelManager", context):
        if context.processor is None:
            context.processor = WhisperProcessor.from_pretrained(
                context.config['environment']['base_model']
            )
        assert context.logger_manager is not None, "LoggerManager must be initialized."
        return ModelManager(
            config=context.config,
            processor=context.processor,
            logger_manager=context.logger_manager
        )

def initialize_data_manager(context: ComponentContext) -> DataManager:
    """Initialize DataManager component."""
    with component_error_handler("DataManager", context):
        if any([
            context.processor is None,
            context.logger_manager is None,
            context.state_manager is None
        ]):
            raise ExperimentError("Processor, LoggerManager, and StateManager must be initialized before DataManager.")
        return DataManager(
            processor=context.processor,
            dataset_name=context.config['dataset']['source'],
            state_manager=context.state_manager,
            logger_manager=context.logger_manager,
            config=context.config
        )

def initialize_trainer(context: ComponentContext) -> TrainerManager:
    """Initialize TrainerManager component."""
    with component_error_handler("TrainerManager", context):
        if not all([
            context.model_manager,
            context.data_manager,
            context.state_manager,
            context.logger_manager
        ]):
            raise ExperimentError("All components must be initialized before TrainerManager.")
        assert isinstance(context.model_manager, ModelManager), "ModelManager must be initialized."
        assert isinstance(context.data_manager, DataManager), "DataManager must be initialized."
        assert isinstance(context.state_manager, StateManager), "StateManager must be initialized."
        assert isinstance(context.logger_manager, LoggerManager), "LoggerManager must be initialized."
        return TrainerManager(
            config=context.config,
            state_manager=context.state_manager,
            model_manager=context.model_manager,
            data_manager=context.data_manager,
            logger_manager=context.logger_manager
        )

def initialize_components(config: Dict[str, Any]) -> Tuple[
    StateManager,
    LoggerManager,
    ModelManager,
    DataManager,
    TrainerManager
]:
    """Initialize all experiment components in the correct order.

    Args:
        config: Configuration dictionary.

    Returns:
        A tuple containing instances of StateManager, LoggerManager, ModelManager, DataManager, and TrainerManager.

    Raises:
        ExperimentError: If any component fails to initialize.
    """
    context = ComponentContext(config=config)

    # Initialize components in dependency order
    context.logger_manager = initialize_logger_manager(context)
    context.state_manager = initialize_state_manager(context)
    context.model_manager = initialize_model_manager(context)
    context.data_manager = initialize_data_manager(context)
    context.trainer = initialize_trainer(context)

    # Verify all components are initialized
    if not all([
        context.state_manager,
        context.logger_manager,
        context.model_manager,
        context.data_manager,
        context.trainer
    ]):
        raise ExperimentError("One or more components failed to initialize.")

    return (
        context.state_manager,
        context.logger_manager,
        context.model_manager,
        context.data_manager,
        context.trainer
    )
