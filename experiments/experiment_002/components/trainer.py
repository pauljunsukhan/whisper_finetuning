"""Training management components"""

import torch
from typing import Dict, Any, Optional, List, Union, cast, TYPE_CHECKING
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    PreTrainedModel,
    EvalPrediction
)
from transformers.training_args import TrainingArguments
from transformers.trainer import TrainerCallback
from torch.nn import Module
from datasets import Dataset
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict  # Updated import


from .base import BaseComponent, ExperimentError
from .state import StateManager
from .model import ModelManager
from .data import DataManager
from .logger import LoggerManager

@dataclass
class MemoryStats:
    """Track GPU memory usage"""
    timestamp: datetime
    allocated: float  # GB
    reserved: float   # GB
    peak_allocated: float = 0.0
    peak_reserved: float = 0.0

class TrainerManager(BaseComponent):
    """Manages training execution"""
    
    def __init__(self, config: Dict[str, Any], state_manager: StateManager, 
                 model_manager: ModelManager, data_manager: DataManager,
                 logger_manager: Optional[LoggerManager] = None):
        super().__init__(config)
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.logger_manager = logger_manager
        self.trainer: Optional[Seq2SeqTrainer] = None
        self._memory_stats: List[MemoryStats] = []
        
        # Validate config values
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate training configurations"""
        required_fields = {
            "training.batch_size": lambda x: x > 0,
            "training.gradient_accumulation_steps": lambda x: x > 0,
            "training.learning_rate": lambda x: x > 0,
            "training.warmup_steps": lambda x: x >= 0,
            "training.max_steps": lambda x: x > 0,
            "training.dataloader_workers": lambda x: x >= 0,
            "training.regularization.label_smoothing": lambda x: 0 <= x <= 1,
            "training.regularization.weight_decay": lambda x: x >= 0,
            "logging.steps.save": lambda x: x > 0,
            "logging.steps.eval": lambda x: x > 0,
            "logging.steps.logging": lambda x: x > 0,
        }
        
        for field, condition in required_fields.items():
            default_values = {
            "training.dataloader_workers": 4,  # Default value as per _get_training_arguments
            # Add more defaults here if necessary
        }

        value = self.get_config(field, default_values.get(field, None))
        if value is None or not condition(value):
            raise ExperimentError(f"Invalid configuration for '{field}': {value}")
    
    def _log_info(self, message: str) -> None:
        """Safe logging helper"""
        if self.logger_manager is not None:
            self.logger_manager.log_info(message)
    
    def _log_error(self, message: str) -> None:
        """Safe error logging helper"""
        if self.logger_manager is not None:
            self.logger_manager.log_error(message)
    
    def _get_memory_stats(self) -> MemoryStats:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return MemoryStats(
                timestamp=datetime.now(),
                allocated=0.0,
                reserved=0.0
            )
            
        stats = MemoryStats(
            timestamp=datetime.now(),
            allocated=torch.cuda.memory_allocated() / 1e9,  # Convert to GB
            reserved=torch.cuda.memory_reserved() / 1e9
        )
        
        # Update peaks
        stats.peak_allocated = max(stats.allocated, getattr(self, '_peak_allocated', 0.0))
        stats.peak_reserved = max(stats.reserved, getattr(self, '_peak_reserved', 0.0))
        self._peak_allocated = stats.peak_allocated
        self._peak_reserved = stats.peak_reserved
        
        return stats
    
    def _log_memory_stats(self, phase: str) -> None:
        """Log memory usage for current phase"""
        if not self.logger_manager:
            return
            
        stats = self._get_memory_stats()
        self._memory_stats.append(stats)
        
        if self.state_manager.is_main_process():
            self._log_info(
                f"Memory usage ({phase}) - "
                f"Allocated: {stats.allocated:.2f}GB, "
                f"Reserved: {stats.reserved:.2f}GB, "
                f"Peak Allocated: {stats.peak_allocated:.2f}GB, "
                f"Peak Reserved: {stats.peak_reserved:.2f}GB"
            )
    
    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """Wrapper for compute_metrics to handle type conversion"""
        return self.model_manager.compute_metrics(eval_preds)
    
    def _get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Get training arguments with safe defaults"""
        try:
            output_dir = str(self.ensure_dir(self.get_config("logging.output_dir")))
            
            return Seq2SeqTrainingArguments(
                output_dir=output_dir,
                logging_dir=self.get_config("logging.log_dir", "logs"),
                per_device_train_batch_size=self.get_config("training.batch_size"),
                per_device_eval_batch_size=2,  # Small eval batch size for stability
                gradient_accumulation_steps=self.get_config("training.gradient_accumulation_steps", 2),
                learning_rate=float(self.get_config("training.learning_rate", 1e-5)),
                warmup_steps=self.get_config("training.warmup_steps", 500),
                max_steps=self.get_config("training.max_steps", 4000),
                # Precision settings - use FP32
                fp16=False,  # Disable mixed precision
                fp16_full_eval=False,  # No FP16 for eval
                bf16=False,  # No bfloat16
                # Training settings
                gradient_checkpointing=self.get_config("training.gradient_checkpointing", False),
                logging_steps=self.get_config("logging.steps.logging", 10),
                save_steps=self.get_config("logging.steps.save", 1000),
                eval_steps=self.get_config("logging.steps.eval", 1000),
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
                push_to_hub=False,
                # Generation settings
                predict_with_generate=True,
                generation_max_length=self.get_config("training.generation_max_length", 225),
                # Column handling
                remove_unused_columns=False,
                label_names=["labels"],
                # Regularization
                weight_decay=self.get_config("training.regularization.weight_decay", 0.01),
                max_grad_norm=1.0,
                label_smoothing_factor=self.get_config("training.regularization.label_smoothing", 0.0),
                # Other
                seed=42,
                dataloader_num_workers=self.get_config("training.dataloader_workers", 4),
                dataloader_pin_memory=True
            )
        except Exception as e:
            raise ExperimentError(f"Failed to create training arguments: {str(e)}")
            
    def train(self) -> None:
        """Run training loop
        
        Raises:
            ExperimentError: If training fails
        """
        try:
            # Log initial memory state
            if torch.cuda.is_available():
                self._log_memory_stats("start")
            
            # Initialize training arguments
            training_args = self._get_training_arguments()
            
            # Create Trainer instance
            model = cast(Optional[PreTrainedModel], self.model_manager.model)
            if model is None:
                raise ExperimentError("Model not initialized")
                
            # Get datasets
            train_dataset = self.data_manager.train_dataset
            eval_dataset = self.data_manager.test_dataset
            
            if train_dataset is None or eval_dataset is None:
                raise ExperimentError("Train or Eval dataset not found.")
            
            # Create trainer with proper type handling
            self.trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,  # type: ignore
                eval_dataset=eval_dataset,    # type: ignore
                data_collator=self.data_manager.collator,
                compute_metrics=self._compute_metrics,
                callbacks=[self._memory_callback()]
            )
            
            # Log memory state after initialization
            if torch.cuda.is_available():
                self._log_memory_stats("after_init")
            
            # Run baseline evaluation
            self._log_info("Running baseline evaluation...")
            try:
                # Ensure eval mode and no gradients for baseline eval
                model.eval()
                with torch.no_grad():
                    if self.trainer is None:
                        raise ExperimentError("Trainer not initialized")
                    baseline_metrics = self.trainer.evaluate()
                    if self.state_manager.is_main_process():
                        self._log_info(f"Baseline metrics: {baseline_metrics}")
            except Exception as e:
                self._log_error(f"Baseline evaluation failed: {str(e)}")
                raise ExperimentError(f"Baseline evaluation failed: {str(e)}") from e
            
            # Start training
            self._log_info("Starting training...")
            if self.trainer is None:
                raise ExperimentError("Trainer not initialized")
            self.trainer.train()
            
            # Final evaluation on main process only
            if self.state_manager.is_main_process():
                self._log_info("Running final evaluation...")
                with torch.no_grad():
                    if self.trainer is None:
                        raise ExperimentError("Trainer not initialized")
                    final_metrics = self.trainer.evaluate()
                    self._log_info(f"Final metrics: {final_metrics}")
            
        except Exception as e:
            self._log_error(f"Training failed: {str(e)}")
            raise ExperimentError(f"Training failed: {str(e)}") from e
    
    def _memory_callback(self) -> TrainerCallback:
        """Create callback for memory monitoring"""
        from transformers.trainer_callback import TrainerCallback
        
        class MemoryMonitorCallback(TrainerCallback):
            def __init__(self, trainer_manager):
                self.tm = trainer_manager
            
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 100 == 0:  # Log every 100 steps
                    self.tm._log_memory_stats(f"step_{state.global_step}")
            
            def on_evaluate(self, args, state, control, **kwargs):
                self.tm._log_memory_stats("evaluation")
        
        return MemoryMonitorCallback(self)
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics
        
        Returns:
            Dictionary of evaluation metrics
        
        Raises:
            ExperimentError: If trainer is not initialized
        """
        if not self.trainer:
            raise ExperimentError("Trainer not initialized")
        return self.trainer.evaluate()
