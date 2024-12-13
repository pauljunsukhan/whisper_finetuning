"""Model management components"""

import torch
import evaluate
from dataclasses import dataclass
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    PreTrainedModel,
    EvalPrediction
)
from typing import Dict, Any, Optional, List, Union, cast, Any
from torch.cuda import Event  # Updated import without aliasing
from .logger import LoggerManager
from .base import BaseComponent
from .data import TypedWhisperProcessor  # Import the Protocol we defined
import numpy as np


@dataclass
class ModelManager:
    """Manages model and processor"""
    config: Dict[str, Any]
    logger_manager: LoggerManager  # Ensure this is passed during initialization
    processor: Optional[TypedWhisperProcessor] = None
    model: Optional[WhisperForConditionalGeneration] = None
    wer_metric: Optional[Any] = None
    
    def __post_init__(self) -> None:
        model_name = self.config["environment"]["base_model"]
        self.processor = cast(TypedWhisperProcessor, WhisperProcessor.from_pretrained(model_name))
        self.model = cast(WhisperForConditionalGeneration, WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP16 instead of FP32
            low_cpu_mem_usage=True
        ))
        
        # Initialize WER metric
        self.wer_metric = evaluate.load("wer")
        
        # Enable gradient checkpointing if configured
        if self.model is not None and self.config["training"].get("optimization", {}).get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False
        
        # Try to enable Flash Attention 2
        if self.model is not None and hasattr(self.model, 'config'):
            try:
                self.model.config.attn_implementation = "flash_attention_2"
            except Exception:
                self.logger_manager.log_error("Flash Attention 2 is not available.")
        
        # **Set a distinct pad token if it's the same as EOS token**
        if self.processor.tokenizer.pad_token == self.processor.tokenizer.eos_token:
            try:
                self.processor.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                self.model.resize_token_embeddings(len(self.processor.tokenizer))
                self.logger_manager.log_info(f"Pad token set to: {self.processor.tokenizer.pad_token}")
            except Exception as e:
                self.logger_manager.log_error(f"Failed to set pad token: {e}")

    def compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """Compute WER metric."""
        pred_ids, label_ids = eval_preds.predictions, eval_preds.label_ids

        # Debugging: Log types and sample data
        self.logger_manager.log_info(f"Type of pred_ids: {type(pred_ids)}")
        self.logger_manager.log_info(f"Type of label_ids: {type(label_ids)}")
        if isinstance(pred_ids, str):
            self.logger_manager.log_info(f"Sample pred_ids: {pred_ids[:50]}")  # Log first 50 chars
        if isinstance(label_ids, str):
            self.logger_manager.log_info(f"Sample label_ids: {label_ids[:50]}")  # Log first 50 chars

        # Ensure processor and wer_metric are initialized
        if self.processor is None or self.wer_metric is None:
            raise ValueError("Processor and WER metric must be initialized.")

        # Check if pred_ids is a string or list of strings
        if isinstance(pred_ids, str):
            # Single string prediction
            pred_str = pred_ids
            label_str = label_ids
        elif isinstance(pred_ids, list) and all(isinstance(p, str) for p in pred_ids):
            # List of string predictions
            pred_str = pred_ids
            label_str = label_ids
        else:
            # Assume pred_ids are token IDs (torch.Tensor or np.ndarray)
            # Convert to torch.Tensor if they are numpy arrays
            if isinstance(pred_ids, np.ndarray):
                pred_ids = torch.from_numpy(pred_ids)
            if isinstance(label_ids, np.ndarray):
                label_ids = torch.from_numpy(label_ids)

            # Move tensors to CPU if needed
            if isinstance(pred_ids, torch.Tensor) and pred_ids.device.type != "cpu":
                pred_ids = pred_ids.cpu()
            if isinstance(label_ids, torch.Tensor) and label_ids.device.type != "cpu":
                label_ids = label_ids.cpu()

            # Replace -100 with pad token id
            if isinstance(label_ids, torch.Tensor):
                pad_token_id = self.processor.tokenizer.pad_token_id
                if pad_token_id is None:
                    raise ValueError("pad_token_id is not set in the tokenizer.")
                pad_token_id = cast(int, pad_token_id)
                label_ids = label_ids.clone()  # To avoid modifying the original data
                label_ids[label_ids == -100] = pad_token_id
            elif isinstance(label_ids, list):
                pad_token_id = self.processor.tokenizer.pad_token_id
                if pad_token_id is None:
                    raise ValueError("pad_token_id is not set in the tokenizer.")
                label_ids = [
                    (id_ if id_ != -100 else pad_token_id)
                    for id_ in label_ids
                ]

            # Convert tensors to lists of lists for batch_decode
            if isinstance(pred_ids, torch.Tensor):
                pred_ids = pred_ids.tolist()
            if isinstance(label_ids, torch.Tensor):
                label_ids = label_ids.tolist()

            # Ensure pred_ids and label_ids are lists of lists
            if isinstance(pred_ids, list) and all(isinstance(p, list) for p in pred_ids):
                pass
            else:
                pred_ids = [pred_ids] if isinstance(pred_ids, list) else [[p] for p in pred_ids]

            if isinstance(label_ids, list) and all(isinstance(l, list) for l in label_ids):
                pass
            else:
                label_ids = [label_ids] if isinstance(label_ids, list) else [[l] for l in label_ids]

            # Type casting to inform the type checker
            pred_ids = cast(List[List[int]], pred_ids)
            label_ids = cast(List[List[int]], label_ids)

            # Decode predictions and labels
            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # Ensure pred_str and label_str are lists of strings
        if isinstance(pred_str, str):
            pred_str = [pred_str]
        if isinstance(label_str, str):
            label_str = [label_str]

        # Compute WER
        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {
            "wer": wer
        }



class MemoryTracker(TrainerCallback):
    """Tracks GPU memory usage and resource utilization"""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger_manager = logger_manager
        self.peak_memory: float = 0
        self.start_time: Optional[Any] = None  # Updated type hint
    
    def on_train_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Record training start time and initial memory state"""
        if torch.cuda.is_available():
            self.start_time = Event(enable_timing=True)  # Use Event directly
            if self.start_time is not None:
                self.start_time.record()
            torch.cuda.reset_peak_memory_stats()
    
    def on_step_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        if state.global_step % 100 == 0:  # Every 100 steps
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            self.peak_memory = max(self.peak_memory, peak)
            
            self.logger_manager.log_info(
                f"Step {state.global_step} memory usage:\n"
                f"  Allocated: {allocated:.2f}GB\n"
                f"  Reserved: {reserved:.2f}GB\n"
                f"  Peak: {peak:.2f}GB"
            )
