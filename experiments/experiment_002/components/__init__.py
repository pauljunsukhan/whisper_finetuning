"""
Core components for the experiment.

This package contains the main components used in the experiment:
- Base: Core base classes and utilities
- Logger: Handles all logging operations
- DataManager: Manages dataset operations
- ModelManager: Handles model training and evaluation
"""

from typing import Dict, Any, Tuple, Optional, cast
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
from datasets import Dataset
import contextlib

from .base import ExperimentError
from .state import StateManager
from .logger import LoggerManager
from .data import DataManager, TypedWhisperProcessor
from .model import ModelManager, MemoryTracker
from .trainer import TrainerManager

ComponentTuple = Tuple[StateManager, LoggerManager, ModelManager, DataManager, TrainerManager]

@dataclass(frozen=False)
class ComponentContext:
    """Context for component initialization"""
    config: dict
    rank: int = field(default=0)
    world_size: int = field(default=1)
    state_manager: Optional[StateManager] = field(default=None)
    logger_manager: Optional[LoggerManager] = field(default=None)
    model_manager: Optional[ModelManager] = field(default=None)
    data_manager: Optional[DataManager] = field(default=None)
    trainer: Optional[TrainerManager] = field(default=None)
    processor: Optional[TypedWhisperProcessor] = field(default=None)

def create_compute_metrics(processor: TypedWhisperProcessor):
    """Create compute_metrics function with access to processor."""
    tokenizer = processor.tokenizer
    def compute_metrics(pred):
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
def component_error_handler(component_name: str, context: Any):
    """Context manager to handle errors in component initialization."""
    try:
        yield
    except Exception as e:
        if context.logger_manager:
            context.logger_manager.log_error(f"{component_name} initialization failed: {str(e)}")
        raise

def initialize_state_manager(context: ComponentContext) -> StateManager:
    """Initialize state manager"""
    with component_error_handler("StateManager", context):
        if context.logger_manager is None:
            raise ExperimentError("Logger manager must be initialized before state manager")
        return StateManager(rank=context.rank, world_size=context.world_size, logger_manager=context.logger_manager)

def initialize_logger_manager(context: ComponentContext) -> LoggerManager:
    """Initialize logger manager"""
    return LoggerManager(config=context.config, rank=context.rank)

def initialize_model_manager(context: ComponentContext) -> ModelManager:
    """Initialize model manager"""
    with component_error_handler("ModelManager", context):
        if context.processor is None:
            processor = WhisperProcessor.from_pretrained(context.config['environment']['base_model'])
            # WhisperProcessor implements TypedWhisperProcessor protocol
            context.processor = cast(TypedWhisperProcessor, processor)
        assert isinstance(context.processor, TypedWhisperProcessor)
        assert context.logger_manager is not None, "logger_manager must be initialized"
        return ModelManager(
            config=context.config,
            processor=context.processor,
            logger_manager=context.logger_manager
        )

def initialize_data_manager(context: ComponentContext) -> DataManager:
    """Initialize data manager"""
    with component_error_handler("DataManager", context):
        if context.processor is None or context.logger_manager is None or context.state_manager is None:
            raise ExperimentError("Required dependencies not initialized")
        assert isinstance(context.processor, WhisperProcessor)
        return DataManager(
            processor=context.processor,
            dataset_name=context.config['dataset']['source'],
            logger_manager=context.logger_manager,
            state_manager=context.state_manager,
            config=context.config  # Added 'config' parameter
        )

def initialize_trainer(context: ComponentContext) -> TrainerManager:
    """Initialize trainer manager"""
    with component_error_handler("Trainer", context):
        if not all([context.model_manager, context.data_manager, context.state_manager, context.logger_manager]):
            raise ExperimentError("Required components not initialized")
        assert isinstance(context.model_manager, ModelManager)
        assert isinstance(context.data_manager, DataManager)
        assert isinstance(context.state_manager, StateManager)
        assert isinstance(context.logger_manager, LoggerManager)
        return TrainerManager(
            config=context.config,
            state_manager=context.state_manager,       # Reordered to match __init__ signature
            model_manager=context.model_manager,
            data_manager=context.data_manager,
            logger_manager=context.logger_manager
        )

def initialize_components(config: dict) -> ComponentTuple:
    """Initialize all components"""
    context = ComponentContext(config=config)
    
    # Initialize in dependency order
    context.logger_manager = initialize_logger_manager(context)
    context.state_manager = initialize_state_manager(context)
    context.model_manager = initialize_model_manager(context)
    context.data_manager = initialize_data_manager(context)
    context.trainer = initialize_trainer(context)
    
    if not all([
        context.state_manager,
        context.logger_manager,
        context.model_manager,
        context.data_manager,
        context.trainer
    ]):
        raise ExperimentError("Component initialization failed")
        
    return (
        context.state_manager,
        context.logger_manager,
        context.model_manager,
        context.data_manager,
        context.trainer
    )
