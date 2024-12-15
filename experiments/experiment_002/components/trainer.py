# File: components/trainer.py

"""Training management components"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, cast
from dataclasses import dataclass
from datetime import datetime
import traceback
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    EvalPrediction,
    GenerationConfig,
    EarlyStoppingCallback,
    TrainingArguments
)
from transformers.trainer import TrainerCallback
from datasets import Dataset
import math
import numpy as np

from .base import BaseComponent, ExperimentError
from .state import StateManager
from .model import ModelManager
from .data import DataManager
from .logger import LoggerManager


@dataclass
class MemoryStats:
    """Track GPU memory usage."""
    timestamp: datetime
    allocated: float  # GB
    reserved: float   # GB
    peak_allocated: float = 0.0
    peak_reserved: float = 0.0


class MemoryMonitor:
    """Handles memory monitoring and statistics."""
    
    @staticmethod
    def get_stats() -> MemoryStats:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return MemoryStats(timestamp=datetime.now(), allocated=0.0, reserved=0.0)
            
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9    # GB
        peak_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9    # GB
        
        torch.cuda.reset_peak_memory_stats()
        
        return MemoryStats(
            timestamp=datetime.now(),
            allocated=allocated,
            reserved=reserved,
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved
        )


class MemoryMonitorCallback(TrainerCallback):
    """TrainerCallback to log GPU memory usage during training."""
    
    def __init__(self, trainer_manager: 'TrainerManager', log_frequency: int = 100):
        self.trainer_manager = trainer_manager
        self.log_frequency = log_frequency
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_frequency == 0:
            self.trainer_manager.log_memory_stats(f"step_{state.global_step}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        self.trainer_manager.log_memory_stats("evaluation")


class ModelDebugger:
    """Handles model debugging and output analysis."""
    
    def __init__(self, logger_manager: LoggerManager, model_manager: ModelManager):
        self.logger_manager = logger_manager
        self.model_manager = model_manager
    
    def debug_model_outputs(self, outputs: Any, inputs: Optional[Dict] = None, prefix: str = "") -> None:
        """Debug model outputs during training/evaluation."""
        if not self.logger_manager:
            return
            
        self.logger_manager.log_info(f"\nDEBUG: {prefix} Model Outputs:")
        self.logger_manager.log_info("-" * 50)
        
        if inputs is not None:
            self._debug_inputs(inputs)
        self._debug_outputs(outputs)
        self.logger_manager.log_info("-" * 50)
    
    def _debug_inputs(self, inputs: Dict) -> None:
        """Debug model input structure."""
        self.logger_manager.log_info("Input structure:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                self._debug_tensor(k, v)
            else:
                self.logger_manager.log_info(f"- {k}: type={type(v)}")
    
    def _debug_outputs(self, outputs: Any) -> None:
        """Debug model output structure."""
        self.logger_manager.log_info("\nOutput structure:")
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    self._debug_tensor(k, v)
                else:
                    self.logger_manager.log_info(f"- {k}: type={type(v)}")
        else:
            self.logger_manager.log_info(f"Output type: {type(outputs)}")
    
    def _debug_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Debug a single tensor's properties."""
        self.logger_manager.log_info(f"- {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        
        if name == "input_features":
            self._debug_input_features(tensor)
        elif name in ["decoder_input_ids", "labels"]:
            self._debug_token_ids(tensor)
        elif name == "logits":
            self._debug_logits(tensor)
        else:
            self.logger_manager.log_info(f"  First values: {tensor[0][:10].tolist()}")
    
    def _debug_input_features(self, tensor: torch.Tensor) -> None:
        """Debug audio input features."""
        self.logger_manager.log_info("  Input features statistics:")
        self.logger_manager.log_info(f"    Overall - min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}")
        
        first_frame = tensor[0, 0]
        mid_frame = tensor[0, tensor.shape[1]//2]
        last_frame = tensor[0, -1]
        
        for name, frame in [("First", first_frame), ("Middle", mid_frame), ("Last", last_frame)]:
            self.logger_manager.log_info(f"    {name} frame - min: {frame.min().item():.4f}, max: {frame.max().item():.4f}, mean: {frame.mean().item():.4f}")
        
        padding_value = -0.61962890625
        non_padding = (tensor[0] != padding_value).any(dim=-1).sum()
        self.logger_manager.log_info(f"    Non-padding frames: {non_padding} out of {tensor.shape[1]}")
    
    def _debug_token_ids(self, tensor: torch.Tensor) -> None:
        """Debug token IDs."""
        tokenizer = self.model_manager.processor.tokenizer
        tokens = [tokenizer.convert_ids_to_tokens(int(id)) for id in tensor[0][:10]]
        self.logger_manager.log_info(f"  First sequence tokens: {tokens}")
    
    def _debug_logits(self, tensor: torch.Tensor) -> None:
        """Debug model logits."""
        self.logger_manager.log_info(f"  Logits stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")


class TrainingArgumentsBuilder:
    """Handles creation and configuration of training arguments."""
    
    def __init__(self, config: Dict[str, Any], logger_manager: LoggerManager, model_manager: ModelManager):
        self.config = config
        self.logger_manager = logger_manager
        self.model_manager = model_manager
        self._validate_and_set_defaults()
    
    def _validate_and_set_defaults(self) -> None:
        """Validate and set default values in config."""
        required_fields = {
            # Environment settings
            "environment.base_model": lambda x: isinstance(x, str) and len(x) > 0,
            # Training settings
            "training.batch_size": lambda x: isinstance(x, int) and x > 0,
            "training.gradient_accumulation_steps": lambda x: isinstance(x, int) and x > 0,
            "training.learning_rate": lambda x: isinstance(x, float) and x > 0,
            "training.warmup_steps": lambda x: isinstance(x, int) and x >= 0,
            "training.max_steps": lambda x: x is None or (isinstance(x, int) and x > 0),  # Allow None for no limit
            "training.dataloader_workers": lambda x: isinstance(x, int) and x >= 0,
            "training.regularization.label_smoothing": lambda x: isinstance(x, float) and 0 <= x <= 1,
            "training.regularization.weight_decay": lambda x: isinstance(x, float) and x >= 0,
            "training.generation_max_length": lambda x: isinstance(x, int) and x > 0,
            # Logging settings
            "logging.steps.save": lambda x: isinstance(x, int) and x > 0,
            "logging.steps.eval": lambda x: isinstance(x, int) and x > 0,
            "logging.steps.logging": lambda x: isinstance(x, int) and x > 0,
            "logging.output_dir": lambda x: x is None or isinstance(x, str),  # Allow None for auto-generation
        }
        
        default_values = {
            # Environment settings
            "environment.base_model": "openai/whisper-small",  # Default to small model
            # Training settings
            "training.batch_size": 16,
            "training.gradient_accumulation_steps": 2,
            "training.learning_rate": 1e-4,
            "training.warmup_steps": 200,
            "training.max_steps": None,  # No limit by default
            "training.dataloader_workers": 16,
            "training.regularization.label_smoothing": 0.1,
            "training.regularization.weight_decay": 0.01,
            "training.generation_max_length": 225,
            # Logging settings
            "logging.steps.save": 100,
            "logging.steps.eval": 50,
            "logging.steps.logging": 25,
            # Generate default output directory
            "logging.output_dir": f"outputs/whisper_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
        
        # First pass: Set all default values if not present
        for field, default_value in default_values.items():
            current_value = self._get_config(field)
            if current_value is None:
                # Split the field into parts
                parts = field.split('.')
                # Navigate to the correct nested level
                config_ref = self.config
                for part in parts[:-1]:
                    if part not in config_ref:
                        config_ref[part] = {}
                    config_ref = config_ref[part]
                # Set the value
                config_ref[parts[-1]] = default_value
                self.logger_manager.log_info(f"Using default value for {field}: {default_value}")
        
        # Second pass: Validate all fields
        for field, condition in required_fields.items():
            value = self._get_config(field)
            if value is None and field != "training.max_steps":  # Allow None for max_steps
                raise ExperimentError(f"Missing required configuration for '{field}'")
            if value is not None and not condition(value):
                raise ExperimentError(f"Invalid configuration value for '{field}': {value}")
                
        # Log all configuration values after validation
        self.logger_manager.log_info("\nValidated Configuration Values:")
        self.logger_manager.log_info("-" * 50)
        for field in required_fields.keys():
            self.logger_manager.log_info(f"{field}: {self._get_config(field)}")
        self.logger_manager.log_info("-" * 50)
    
    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def _ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def build(self) -> Seq2SeqTrainingArguments:
        """Build training arguments."""
        try:
            self._debug_config()
            output_dir = self._get_config("logging.output_dir", "outputs")
            batch_size = self._get_config("training.batch_size", 16)
            dataloader_workers = self._get_config("training.dataloader_workers", 16)

            training_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=16,
                weight_decay=0.0,
                gradient_accumulation_steps=1,
                learning_rate=1e-5,
                warmup_steps=500,
                max_steps=1000,
                gradient_checkpointing=False,
                max_grad_norm=None,  # Gradient clipping
                fp16=True,
                evaluation_strategy="steps",
                eval_steps=20,  # Frequent evaluation
                save_steps=20,  # Align with eval_steps
                logging_steps=20,
                report_to=["tensorboard"],
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
                remove_unused_columns=False,
                generation_max_length=150,  # Match with generation config
                predict_with_generate=True,
                dataloader_num_workers=dataloader_workers,
                save_total_limit=3,
                eval_accumulation_steps=1,
                fp16_full_eval=False,  # Avoid potential numerical issues
                label_smoothing_factor=0.0,  # Correct parameter name
                
            )

            return training_args
            
        except Exception as e:
            self.logger_manager.log_error(f"Failed to create training arguments: {str(e)}")
            self.logger_manager.log_error(f"Full error traceback:\n{traceback.format_exc()}")
            raise ExperimentError(f"Failed to create training arguments: {str(e)}") from e
    
    def _debug_config(self) -> None:
        """Debug configuration values."""
        self.logger_manager.log_info("\nDEBUG: Training Arguments Configuration:")
        self.logger_manager.log_info("-" * 50)
        for key in ["logging.output_dir", "logging.log_dir", "training.batch_size", 
                   "training.learning_rate", "training.max_steps"]:
            self.logger_manager.log_info(f"{key}: {self._get_config(key)}")
        self.logger_manager.log_info("-" * 50)
    
    def _create_generation_config(self) -> GenerationConfig:
        config = GenerationConfig()
        
        # Let Whisper handle tokens
        config.forced_decoder_ids = self.model_manager.processor.get_decoder_prompt_ids(
            language="en",
            task="transcribe"
        )
        
        config.max_new_tokens = 150
        config.use_cache = False
        
        return config
    
    def _debug_generation_config(self, config: GenerationConfig) -> None:
        """Debug generation configuration."""
        self.logger_manager.log_info("\nDEBUG: Generation Config:")
        self.logger_manager.log_info("-" * 50)
        for key in ["max_new_tokens", "num_beams", "do_sample", "use_cache", "forced_decoder_ids"]:
            self.logger_manager.log_info(f"{key}: {getattr(config, key)}")
        self.logger_manager.log_info("-" * 50)
    


class TrainerManager(BaseComponent):
    """Manages training execution."""
    
    def __init__(self, config: Dict[str, Any], state_manager: StateManager, 
                 model_manager: ModelManager, data_manager: DataManager,
                 logger_manager: LoggerManager):
        """Initialize TrainerManager with necessary components."""
        super().__init__(config)
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.logger_manager = logger_manager
        self.trainer: Optional[Seq2SeqTrainer] = None
        self._memory_stats: List[MemoryStats] = []
        
        self.memory_monitor = MemoryMonitor()
        self.model_debugger = ModelDebugger(logger_manager, model_manager)
        self.training_args_builder = TrainingArgumentsBuilder(config, logger_manager, model_manager)
    
    def log_memory_stats(self, phase: str) -> None:
        """Log memory usage for current phase."""
        if not self.state_manager.is_main_process():
            return
            
        stats = self.memory_monitor.get_stats()
        self._memory_stats.append(stats)
        
        self.logger_manager.log_info(
            f"Memory usage ({phase}) - "
            f"Allocated: {stats.allocated:.2f}GB, "
            f"Reserved: {stats.reserved:.2f}GB, "
            f"Peak Allocated: {stats.peak_allocated:.2f}GB, "
            f"Peak Reserved: {stats.peak_reserved:.2f}GB"
        )
    
    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if self.state_manager.is_main_process():
            # Log memory before generation
            self.log_memory_stats("before_generation")
            
            self._debug_predictions(eval_preds)
            
            # Log memory after generation
            self.log_memory_stats("after_generation")
            
            # Force CUDA cache clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.log_memory_stats("after_cache_clear")
        
        metrics = self.model_manager.compute_metrics(eval_preds)
        if self.state_manager.is_main_process():
            self.logger_manager.log_info(f"Computed metrics: {metrics}")
        return metrics
    
    def _debug_predictions(self, eval_preds: EvalPrediction) -> None:
        """Debug prediction structure and content."""
        self.logger_manager.log_info("\nDEBUG: Prediction Structure:")
        self.logger_manager.log_info("-" * 50)
        self.logger_manager.log_info(f"Predictions shape: {eval_preds.predictions.shape if hasattr(eval_preds.predictions, 'shape') else 'No shape'}")
        self.logger_manager.log_info(f"Labels shape: {eval_preds.label_ids.shape if hasattr(eval_preds.label_ids, 'shape') else 'No shape'}")
        
        try:
            self._debug_decoded_examples(eval_preds)
            self._debug_tokenizer_settings()
        except Exception as e:
            self.logger_manager.log_error(f"Error during decoding: {str(e)}")
            self.logger_manager.log_error(f"Full decoding error traceback:\n{traceback.format_exc()}")
    
    def _debug_decoded_examples(self, eval_preds: EvalPrediction) -> None:
        """Debug decoded prediction examples."""
        decoded_preds = self.model_manager.processor.batch_decode(eval_preds.predictions, skip_special_tokens=True)
        decoded_labels = self.model_manager.processor.batch_decode(eval_preds.label_ids, skip_special_tokens=True)
        
        self.logger_manager.log_info("\nDEBUG: Decoded Examples:")
        self.logger_manager.log_info("-" * 50)
        for i in range(min(3, len(decoded_preds))):
            self.logger_manager.log_info(f"\nExample {i+1}:")
            self.logger_manager.log_info(f"Prediction tokens: {eval_preds.predictions[i][:20]}")
            self.logger_manager.log_info(f"Label tokens: {eval_preds.label_ids[i][:20]}")
            self.logger_manager.log_info(f"Decoded prediction: '{decoded_preds[i]}'")
            self.logger_manager.log_info(f"Decoded reference: '{decoded_labels[i]}'")
    
    def _debug_tokenizer_settings(self) -> None:
        """Debug tokenizer configuration."""
        tokenizer = self.model_manager.processor.tokenizer
        self.logger_manager.log_info("\nDEBUG: Tokenizer Settings:")
        self.logger_manager.log_info("-" * 50)
        
        # Get Whisper's actual special tokens
        start_token = "<|startoftranscript|>"
        lang_token = "<|en|>"
        task_token = "<|transcribe|>"
        notimestamps_token = "<|notimestamps|>"
        
        # Convert to IDs
        start_id = tokenizer.convert_tokens_to_ids(start_token)
        lang_id = tokenizer.convert_tokens_to_ids(lang_token)
        task_id = tokenizer.convert_tokens_to_ids(task_token)
        notimestamps_id = tokenizer.convert_tokens_to_ids(notimestamps_token)
        
        # Log actual Whisper tokens
        self.logger_manager.log_info(f"Start token: {start_token} ({start_id})")
        self.logger_manager.log_info(f"Language token: {lang_token} ({lang_id})")
        self.logger_manager.log_info(f"Task token: {task_token} ({task_id})")
        self.logger_manager.log_info(f"No timestamps token: {notimestamps_token} ({notimestamps_id})")
        
        # Show token sequence for clarity
        forced_decoder_ids = self.model_manager.processor.get_decoder_prompt_ids(language="en", task="transcribe")
        self.logger_manager.log_info("\nForced decoder sequence:")
        for pos, token_id in forced_decoder_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            self.logger_manager.log_info(f"Position {pos}: {token} ({token_id})")
        
        self.logger_manager.log_info("\nOther special tokens:")
        self.logger_manager.log_info(f"Pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        self.logger_manager.log_info(f"End token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    
    def train(self) -> None:
        """Train the model using the prepared datasets."""
        try:
            training_args = self.training_args_builder.build()

            # Create and apply generation config directly to the model
            generation_config = self.training_args_builder._create_generation_config()
            self.model_manager.model.config.update(generation_config.to_dict())
            self.model_manager.model.config.forced_decoder_ids = generation_config.forced_decoder_ids
            self.model_manager.model.config.decoder_start_token_id = generation_config.decoder_start_token_id
            self.model_manager.model.config.bos_token_id = generation_config.bos_token_id
            self.model_manager.model.config.eos_token_id = generation_config.eos_token_id
            self.model_manager.model.config.pad_token_id = generation_config.pad_token_id
            self.model_manager.model.config.use_cache = generation_config.use_cache
            self.model_manager.model.config.max_length = generation_config.max_new_tokens

            if not self.data_manager.datasets:
                raise ExperimentError("Datasets not prepared. Call prepare_datasets first.")
            
            train_dataset = self.data_manager.datasets.get('train')
            eval_dataset = self.data_manager.datasets.get('test')
            
            if train_dataset is None or eval_dataset is None:
                raise ExperimentError("Train or eval dataset not found in DataManager")
            
            # Add all callbacks including the new gradient logging
            callbacks = [
                MemoryMonitorCallback(self, log_frequency=self.get_config("logging.memory_log_interval", 100)),
                GradientLoggingCallback(self, log_frequency=self.get_config("logging.gradient_log_interval", 100)),
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01
                )
            ]

            self.trainer = CustomSeq2SeqTrainer(
                model=self.model_manager.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=self.data_manager.collator,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks,
                tokenizer=self.model_manager.processor.tokenizer,
            )
            
            # Add logger manager to trainer for gradient logging
            self.trainer.logger_manager = self.logger_manager

            if self.state_manager.is_main_process():
                self.log_memory_stats("after_init")
            
            # Run baseline evaluation
            self.logger_manager.log_info("Running baseline evaluation...")
            baseline_metrics = self.evaluate()
            self.logger_manager.log_info(f"Baseline metrics: {baseline_metrics}")
            
            self.logger_manager.log_info("Starting training...")
            self.trainer.train()
            
        except Exception as e:
            self.logger_manager.log_error(f"Training failed: {str(e)}")
            self.logger_manager.log_error(f"Full error traceback:\n{traceback.format_exc()}")
            raise ExperimentError(f"Training failed: {str(e)}") from e

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        if not self.trainer:
            raise ExperimentError("Trainer not initialized")
        
        # Set language and task explicitly in model configuration
        self.model_manager.model.config.language = "en"
        self.model_manager.model.config.task = "transcribe"
        
        # Evaluate using the model's updated configuration
        return self.trainer.evaluate()


class GradientLoggingCallback(TrainerCallback):
    """Callback to log gradient statistics during training."""
    
    def __init__(self, trainer_manager: 'TrainerManager', log_frequency: int = 100):
        self.trainer_manager = trainer_manager
        self.log_frequency = log_frequency
        self.logger = trainer_manager.logger_manager
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log gradient statistics after each optimization step."""
        if state.global_step % self.log_frequency == 0:
            model = self.trainer_manager.trainer.model
            grad_stats = {}
            
            # Collect gradient statistics for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    grad_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'norm': grad.norm().item()
                    }
            
            # Log summary statistics
            self.logger.log_info(f"\nGradient Statistics (Step {state.global_step}):")
            self.logger.log_info("-" * 50)
            
            # Calculate overall statistics
            all_means = [stats['mean'] for stats in grad_stats.values()]
            all_norms = [stats['norm'] for stats in grad_stats.values()]
            
            self.logger.log_info(f"Global gradient statistics:")
            self.logger.log_info(f"Mean gradient: {np.mean(all_means):.6f}")
            self.logger.log_info(f"Mean gradient norm: {np.mean(all_norms):.6f}")
            
            # Log statistics for important layers
            important_layers = ['encoder.conv1', 'encoder.conv2', 'decoder.embed_tokens', 'decoder.final_layer_norm']
            for layer_prefix in important_layers:
                matching_layers = {k: v for k, v in grad_stats.items() if any(k.startswith(p) for p in [layer_prefix])}
                if matching_layers:
                    self.logger.log_info(f"\n{layer_prefix} layers:")
                    for name, stats in matching_layers.items():
                        self.logger.log_info(
                            f"{name}:\n"
                            f"  Mean: {stats['mean']:.6f}\n"
                            f"  Std:  {stats['std']:.6f}\n"
                            f"  Min:  {stats['min']:.6f}\n"
                            f"  Max:  {stats['max']:.6f}\n"
                            f"  Norm: {stats['norm']:.6f}"
                        )
            
            self.logger.log_info("-" * 50)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer with gradient logging capabilities."""
    
    def _prepare_inputs_labels_for_model(self, inputs):
        """Prepare inputs and labels, ensuring proper padding masking."""
        # First apply the parent class preparation
        inputs = self._prepare_inputs(inputs)
        
        # Handle label padding
        if "labels" in inputs:
            labels = inputs["labels"].clone()
            # Replace padding tokens with -100 to ignore them in loss computation
            padding_mask = labels == self.tokenizer.pad_token_id
            labels[padding_mask] = -100
            inputs["labels"] = labels
            
        return inputs
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step to ensure proper language and task tokens."""
        model.eval()
        inputs = self._prepare_inputs_labels_for_model(inputs)
        
        has_labels = "labels" in inputs
        with torch.no_grad():
            if has_labels:
                labels = inputs["labels"]
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                labels = None
                loss = None

            generated_tokens = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),  # Pass mask explicitly
                forced_decoder_ids=model.config.forced_decoder_ids,
                max_new_tokens=self.args.generation_max_length,
                language="en",
                task="transcribe",
                use_cache=False
            )
        
        return (loss, generated_tokens, labels)
    
    def training_step(self, model, inputs):
        """Override training step to add gradient logging."""
        # Prepare inputs with proper padding masking
        inputs = self._prepare_inputs_labels_for_model(inputs)
        loss = super().training_step(model, inputs)
        
        # Log gradients before gradient clipping
        if self.args.max_grad_norm is not None and self.state.global_step % 100 == 0:
            self.log_gradients_before_clipping(model)
        
        return loss
    
    def log_gradients_before_clipping(self, model):
        """Log gradient norms before clipping."""
        if not hasattr(self, "logger_manager"):
            return
        
        total_norm = 0.0
        param_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                param_norms[name] = param_norm.item()
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        self.logger_manager.log_info(f"\nGradient norms before clipping (step {self.state.global_step}):")
        self.logger_manager.log_info(f"Total gradient norm: {total_norm:.4f}")
        
        # Log top 5 largest gradients
        sorted_norms = sorted(param_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger_manager.log_info("\nTop 5 largest gradients:")
        for name, norm in sorted_norms:
            self.logger_manager.log_info(f"{name}: {norm:.4f}")

