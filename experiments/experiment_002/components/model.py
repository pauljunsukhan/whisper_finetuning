# File: components/model.py

"""Model management components"""

import torch
import evaluate
from dataclasses import dataclass, field
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    EvalPrediction
)
from typing import Dict, Any, Optional, List, Union
from .base import BaseComponent, ExperimentError
from .logger import LoggerManager
import traceback


@dataclass
class ModelManager(BaseComponent):
    """Manages model and processor"""
    logger_manager: LoggerManager  # Non-default argument must be first
    processor: Optional[WhisperProcessor] = None
    model: Optional[WhisperForConditionalGeneration] = None
    wer_metric: Optional[Any] = None

    def __post_init__(self) -> None:
        """Initialize processor, model, and metrics."""
        self._initialize_cuda()
        self._initialize_processor()
        self._initialize_model()
        self._initialize_metrics()
        self._configure_model()
        self._apply_optimizations()

    def _initialize_cuda(self) -> None:
        """Initialize CUDA settings for optimal performance."""
        try:
            if torch.cuda.is_available():
                # Enable cuDNN benchmarking for better performance
                torch.backends.cudnn.benchmark = True
                self.log_info("CUDA optimizations enabled:")
                self.log_info("- cuDNN benchmark: True")
                self.log_info(f"- CUDA device: {torch.cuda.get_device_name()}")
                self.log_info(f"- CUDA version: {torch.version.cuda}")
                self.log_info(f"- cuDNN version: {torch.backends.cudnn.version()}")
        except Exception as e:
            self.log_error(f"Failed to initialize CUDA settings: {e}")
            # Don't raise error as this is not critical

    def _initialize_processor(self) -> None:
        """Initialize the Whisper processor."""
        try:
            model_name = self.get_config("environment.base_model")
            self.processor = WhisperProcessor.from_pretrained(model_name)
            
            # Configure feature extractor settings
            self.processor.feature_extractor.n_fft = 400  # 25ms at 16kHz
            self.processor.feature_extractor.hop_length = 160  # 10ms at 16kHz
            self.processor.feature_extractor.n_mels = 128  # Whisper Large V3 uses 128 mel bins
            self.processor.feature_extractor.chunk_length = None  # Don't chunk the audio
            self.processor.feature_extractor.sampling_rate = 16000  # Whisper expects 16kHz
            self.processor.feature_extractor.return_attention_mask = True  # Enable attention masks
            self.processor.feature_extractor.feature_size = 1  # Default feature size
            self.processor.feature_extractor.padding_value = 0.0  # Use 0 for padding
            self.processor.feature_extractor.do_normalize = True  # Enable normalization
            self.processor.feature_extractor.dtype = torch.float32  # Always use FP32 for feature extraction
            self.processor.feature_extractor.padding = True  # Enable padding
            self.processor.feature_extractor.padding_side = "right"  # Pad on the right
            self.processor.feature_extractor.return_attention_mask = True  # Get attention masks
            self.processor.feature_extractor.max_length = 3000  # Force padding to 3000 frames
            self.processor.feature_extractor.truncation = True  # Enable truncation
            #self.processor.feature_extractor.mel_filters = None  # Reset mel filters to force recomputation
            #self.processor.feature_extractor.mel_params = None  # Reset mel params to force recomputation
            
            # Force recompute mel filters with correct settings
            _ = self.processor.feature_extractor.mel_filters
            
            self.log_info(f"Processor loaded from {model_name}.")
            self.log_info("\nFeature Extractor Configuration:")
            self.log_info(f"- n_fft: {self.processor.feature_extractor.n_fft}")
            self.log_info(f"- hop_length: {self.processor.feature_extractor.hop_length}")
            self.log_info(f"- n_mels: {self.processor.feature_extractor.n_mels}")
            self.log_info(f"- chunk_length: {self.processor.feature_extractor.chunk_length}")
            self.log_info(f"- sampling_rate: {self.processor.feature_extractor.sampling_rate}")
            self.log_info(f"- do_normalize: {self.processor.feature_extractor.do_normalize}")
            self.log_info(f"- dtype: {self.processor.feature_extractor.dtype}")
            self.log_info(f"- max_length: {self.processor.feature_extractor.max_length}")
            
        except Exception as e:
            self.log_error(f"Failed to load WhisperProcessor: {e}")
            raise ExperimentError(f"Failed to load WhisperProcessor: {e}") from e

    def _initialize_model(self) -> None:
        """Initialize the Whisper model."""
        try:
            model_name = self.get_config("environment.base_model")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize with standard settings
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to GPU immediately if available
            self.model = self.model.to(device)
            
            self.log_info(f"Model loaded from {model_name} with dtype torch.float32")
            self.log_info(f"Model device: {next(self.model.parameters()).device}")
            self.log_info("Note: Mixed precision training will be handled by the trainer if enabled")
            
            # Log model parameters dtype for verification
            if hasattr(self.model, 'parameters'):
                param = next(self.model.parameters())
                self.log_info(f"Model parameters dtype: {param.dtype}")
                
        except Exception as e:
            self.log_error(f"Failed to load WhisperForConditionalGeneration: {e}")
            raise ExperimentError(f"Failed to load WhisperForConditionalGeneration: {e}") from e

    def _get_torch_dtype(self) -> torch.dtype:
        """Determine the torch data type based on configuration."""
        mixed_precision = self.get_config("environment.compute.mixed_precision", "fp16")
        if isinstance(mixed_precision, bool):
            return torch.float16 if mixed_precision else torch.float32
        elif isinstance(mixed_precision, str):
            if mixed_precision.lower() == "fp16":
                return torch.float16
            elif mixed_precision.lower() == "fp32":
                return torch.float32
        self.log_error(f"Unsupported mixed_precision type: {mixed_precision}. Defaulting to fp32.")
        return torch.float32

    def _initialize_metrics(self) -> None:
        """Initialize evaluation metrics."""
        try:
            self.wer_metric = evaluate.load("wer")
            self.log_info("WER metric initialized.")
        except Exception as e:
            self.log_error(f"Failed to load WER metric: {e}")
            raise ExperimentError(f"Failed to load WER metric: {e}") from e

    def _configure_model(self) -> None:
        """Configure model settings like gradient checkpointing and attention implementation."""
        try:
            if self.model is None:
                return

            # Configure model settings in the correct order
            self._configure_generation_settings()
            self._configure_forced_tokens()

            # Enable gradient checkpointing properly
            self.model.config.use_cache = False  # Must be disabled for gradient checkpointing
            if self.get_config("environment.compute.gradient_checkpointing", True):  # Enable by default
                self.model.gradient_checkpointing_enable()
                self.log_info("Gradient checkpointing enabled")
            
            # Set model to training mode
            self.model.train()

            # Log updated model settings
            self._log_model_config()

        except Exception as e:
            self.log_error(f"Failed to configure the model: {e}")
            raise ExperimentError(f"Failed to configure the model: {e}") from e


    def _configure_generation_settings(self) -> None:
        """Configure generation-related model settings."""
        self.model.config.update({
            "max_length": self.get_config("training.generation_max_length", 225),
            "num_beams": 1,  # Greedy decoding during training
            "do_sample": False,  # No sampling during training
            "no_repeat_ngram_size": 2,  # Disable n-gram blocking
            "length_penalty": 1.2,  # No length penalty
            "suppress_tokens": [self.processor.tokenizer.unk_token_id],  # Suppress unlikely outputs
            "dropout": 0.0,  # Aligned with Whisper fine-tuning guidelines
            "is_encoder_decoder": True,  # Ensure encoder-decoder mode
        })


    def _configure_forced_tokens(self) -> None:
        """Configure forced tokens for language and task."""
        language = self.get_config("dataset.features.language", "en")
        task = self.get_config("dataset.features.task", "transcribe")

        # Let the tokenizer handle prefix tokens automatically
        self.processor.tokenizer.set_prefix_tokens(language=language, task=task)
        self.log_info(f"Tokenizer configured with language={language}, task={task}")

        # Remove forced_decoder_ids during training
        self.model.config.forced_decoder_ids = None
        self.log_info("Removed forced_decoder_ids from model configuration during training")

        # Set decoder_start_token_id appropriately
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.log_info(f"Set decoder_start_token_id to {self.model.config.decoder_start_token_id}")

        # Remove explicit token ID settings if any
        self.model.config.update({
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
        })


    def _log_model_config(self) -> None:
        """Log updated model configuration settings."""
        self.log_info("\nDEBUG: Updated Model Settings:")
        self.log_info("-" * 50)
        config_values = {
            "Dropout rate": self.model.config.dropout,
            "Token suppression": self.model.config.suppress_tokens,
            "Max length": self.model.config.max_length,
            "Pad token ID": self.model.config.pad_token_id,
            "BOS token ID": self.model.config.bos_token_id,
            "EOS token ID": self.model.config.eos_token_id,
            "Decoder start token ID": self.model.config.decoder_start_token_id,
            "Forced decoder IDs": self.model.config.forced_decoder_ids,
            "Training mode": self.model.training,
        }
        for key, value in config_values.items():
            self.log_info(f"{key}: {value}")


    def _set_distinct_pad_token(self) -> None:
        """Set a distinct padding token if it matches the EOS token."""
        try:
            new_pad_token = '<|pad|>'
            if new_pad_token not in self.processor.tokenizer.get_vocab():
                self.processor.tokenizer.add_special_tokens({'pad_token': new_pad_token})
                self.model.resize_token_embeddings(len(self.processor.tokenizer))
                self.log_info(f"Pad token set to: {new_pad_token}")
            else:
                self.processor.tokenizer.pad_token = new_pad_token
                self.model.config.pad_token_id = self.processor.tokenizer.convert_tokens_to_ids(new_pad_token)
                self.log_info(f"Pad token already exists. Set pad_token to: {new_pad_token}")
        except Exception as e:
            self.log_error(f"Failed to set distinct pad token: {e}")
            raise ExperimentError(f"Failed to set distinct pad token: {e}") from e

    def compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """Compute WER metric."""
        try:
            pred_ids, label_ids = eval_preds.predictions, eval_preds.label_ids

            # Replace -100 in the labels as we can't decode them
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

            # Decode predictions and references
            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            # Log sample predictions and references
            self.log_info(f"Sample prediction: {pred_str[0]}")
            self.log_info(f"Sample reference: {label_str[0]}")

            # Compute WER
            wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}
        except Exception as e:
            self.log_error(f"Failed to compute metrics: {e}")
            raise ExperimentError(f"Failed to compute metrics: {e}") from e

    def _apply_optimizations(self) -> None:
        """Apply model optimizations like torch.compile."""
        try:
            if self.model is None:
                return

            # Debug optimization state before changes
            self.log_info("\nDEBUG: Pre-optimization Model State:")
            self.log_info("-" * 50)
            self.log_info(f"Model device: {next(self.model.parameters()).device}")
            self.log_info(f"Model dtype: {next(self.model.parameters()).dtype}")
            self.log_info(f"Model training: {self.model.training}")
            self.log_info(f"Model class: {self.model.__class__.__name__}")
            self.log_info(f"Model is compiled: {hasattr(self.model, '_orig_mod')}")

            # Apply torch.compile if enabled
            torch_compile = self.get_config("environment.compute.torch_compile", False)
            compile_mode = self.get_config("environment.compute.compile_mode", "reduce-overhead")
            
            if torch_compile:
                self.log_info(f"\nStarting torch.compile process:")
                self.log_info("-" * 50)
                self.log_info(f"Compilation mode: {compile_mode}")
                self.log_info(f"PyTorch version: {torch.__version__}")
                self.log_info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    self.log_info(f"CUDA device: {torch.cuda.get_device_name()}")
                    self.log_info(f"CUDA capability: {torch.cuda.get_device_capability()}")
                
                try:
                    # Set model to eval mode during compilation
                    training_mode = self.model.training
                    self.model.eval()
                    self.log_info("Model set to eval mode for compilation")
                    
                    # Log compilation settings
                    self.log_info("\nCompilation settings:")
                    self.log_info(f"- Mode: {compile_mode}")
                    self.log_info("- fullgraph: False (avoiding full graph optimization)")
                    self.log_info("- dynamic: True (allowing dynamic shapes)")
                    
                    self.log_info("\nStarting compilation...")
                    # Compile with specific settings
                    self.model = torch.compile(
                        self.model, 
                        mode=compile_mode,
                        fullgraph=False,  # Avoid full graph optimization
                        dynamic=True,  # Allow dynamic shapes
                    )
                    self.log_info("Model compilation completed successfully")
                    
                    # Restore original training mode
                    if training_mode:
                        self.model.train()
                        self.log_info("Model restored to training mode")
                    
                    # Debug final state
                    self.log_info("\nDEBUG: Post-compilation Model State:")
                    self.log_info("-" * 50)
                    self.log_info(f"Model device: {next(self.model.parameters()).device}")
                    self.log_info(f"Model dtype: {next(self.model.parameters()).dtype}")
                    self.log_info(f"Model training: {self.model.training}")
                    self.log_info(f"Model class: {self.model.__class__.__name__}")
                    self.log_info(f"Model is compiled: {hasattr(self.model, '_orig_mod')}")
                    
                except Exception as e:
                    self.log_error(f"Failed to apply torch.compile: {str(e)}")
                    self.log_error(f"Full compilation error traceback:\n{traceback.format_exc()}")
                    self.log_error("Continuing without torch.compile optimization")
            else:
                self.log_info("torch.compile is disabled in configuration")
            
        except Exception as e:
            self.log_error(f"Failed to apply optimizations: {str(e)}")
            self.log_error(f"Full optimization error traceback:\n{traceback.format_exc()}")
            # Don't raise error as optimizations are not critical
