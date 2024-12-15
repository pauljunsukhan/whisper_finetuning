# File: components/data.py

"""Data processing for Whisper fine-tuning."""

from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
from dataclasses import dataclass
from datasets import Dataset, load_dataset
from transformers import WhisperProcessor
from .base import BaseComponent, ExperimentError
from .logger import LoggerManager
from .state import StateManager


@dataclass
class DataCollator:
    """Collates data for Whisper training."""
    processor: WhisperProcessor
    
    def __call__(self, features: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        batch = {
            key: torch.from_numpy(np.stack([feature[key] for feature in features]))
            for key in features[0].keys()
        }

        # Convert to proper dtypes
        batch["input_features"] = batch["input_features"].to(torch.float32)
        batch["labels"] = batch["labels"].to(torch.long)

        # Get sequence dimensions
        batch_size, seq_length = batch["labels"].shape

        # Create decoder input IDs
        batch["decoder_input_ids"] = torch.full(
            (batch_size, seq_length),
            self.processor.tokenizer.pad_token_id,
            dtype=torch.long
        )

        # Get prefix tokens from tokenizer
        prefix_tokens = self.processor.tokenizer.prefix_tokens
        prefix_length = len(prefix_tokens)

        # Apply prefix tokens at the start of each sequence
        batch["decoder_input_ids"][:, :prefix_length] = torch.tensor(prefix_tokens)

        # Teacher forcing: shift target sequence right by one position
        valid_length = seq_length - 1
        if valid_length > prefix_length:
            batch["decoder_input_ids"][:, prefix_length:valid_length] = batch["labels"][:, prefix_length:valid_length].clone()

        # Replace padding token id's of the labels by -100 to ignore them in the loss
        batch["labels"] = batch["labels"].masked_fill(
            batch["labels"] == self.processor.tokenizer.pad_token_id,
            -100
        )

        return batch


@dataclass
class DataManager(BaseComponent):
    """Manages dataset preparation for Whisper fine-tuning."""
    processor: WhisperProcessor
    dataset_name: str
    state_manager: StateManager
    logger_manager: LoggerManager
    datasets: Optional[Dict[str, Dataset]] = None
    collator: Optional[DataCollator] = None
    _last_example: Optional[Dict[str, Any]] = None
    _current_split: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Initialize after instance creation."""
        self.collator = DataCollator(self.processor)
        
        # Configure tokenizer
        language = self.get_config("dataset.features.language", "en")
        task = self.get_config("dataset.features.task", "transcribe")
        self.processor.tokenizer.set_prefix_tokens(language=language, task=task)
        self.log_info(f"Tokenizer configured with language={language}, task={task}")
    
    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process a single example.
        
        Args:
            example: Dictionary containing audio and text data
            
        Returns:
            Dictionary containing preprocessed numpy arrays for model input
        """
        try:
            # Input validation
            if "audio" not in example:
                raise ValueError("Missing 'audio' key in input example")
            if "text" not in example:
                raise ValueError("Missing 'text' key in input example")
            if not isinstance(example["audio"], dict):
                raise ValueError(f"Expected audio to be a dict, got {type(example['audio'])}")
            if "array" not in example["audio"]:
                raise ValueError("Missing 'array' key in audio dict")
            if "sampling_rate" not in example["audio"]:
                raise ValueError("Missing 'sampling_rate' key in audio dict")
            if not isinstance(example["text"], str):
                raise ValueError(f"Expected text to be a string, got {type(example['text'])}")
            
            # Process audio
            audio_array = example['audio']['array']
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            audio_array = audio_array.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))

            # Calculate required audio length for 3000 frames
            n_fft = 400
            hop_length = 160
            target_length = (3000 - 1) * hop_length + n_fft  # Length needed for exactly 3000 frames
            
            # Store original length for attention mask
            original_length = len(audio_array)
            
            # Pad audio if needed
            if len(audio_array) < target_length:
                padding = target_length - len(audio_array)
                audio_array = np.pad(audio_array, (0, padding), mode="constant")
            else:
                # Truncate if longer
                audio_array = audio_array[:target_length]
            
            # Process audio features
            feature_output = self.processor(
                audio_array,
                sampling_rate=16000,  # Dataset uses 16kHz
                return_tensors="pt",
                do_normalize=True,
                n_fft=n_fft,  # 25ms at 16kHz
                hop_length=hop_length,  # 10ms at 16kHz
                n_mels=128,  # Whisper Large V3 uses 128 mel bins
                padding=False,  # No padding needed since we padded the raw audio
                return_attention_mask=False  # We'll create our own mask
            )
            
            # Convert features to numpy
            features = feature_output.input_features[0, :, :3000].numpy()
            
            # Manually create attention mask based on valid frames
            valid_frames = min(original_length, target_length)
            valid_frame_count = (valid_frames - n_fft) // hop_length + 1
            encoder_attention_mask = np.zeros(3000, dtype=np.int64)
            encoder_attention_mask[:valid_frame_count] = 1
            
            # Let the tokenizer handle prefix tokens automatically
            text_tokens = self.processor.tokenizer(
                example["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.get_config("training.max_label_length", 448),
                add_special_tokens=True,  # This ensures all special tokens are added
                return_attention_mask=True
            )
            
            # Convert to lists/numpy for storage
            labels = text_tokens.input_ids[0].numpy()
            decoder_attention_mask = text_tokens.attention_mask[0].numpy()
            
            # Store example info for logging after split completion
            if self.state_manager.is_main_process():
                self._last_example = {
                    'audio_shape': audio_array.shape,
                    'audio_array': audio_array,
                    'text': example["text"],
                    'features_shape': features.shape,
                    'encoder_attention_mask_shape': encoder_attention_mask.shape,
                    'decoder_attention_mask_shape': decoder_attention_mask.shape,
                    'label_length': len(labels)
                }
            
            return {
                "input_features": features,  # numpy [128, 3000]
                "attention_mask": encoder_attention_mask,  # numpy [3000]
                "labels": labels,  # numpy array
                "decoder_attention_mask": decoder_attention_mask  # numpy array
            }
        
        except Exception as e:
            self.log_error(f"Error processing example: {str(e)}")
            if self.state_manager.is_main_process():
                import traceback
                self.log_error(f"Full traceback:\n{traceback.format_exc()}")
            raise ExperimentError(f"Failed to process example: {str(e)}") from e
    
    def prepare_datasets(self, test_size: float = 0.2, seed: Optional[int] = None) -> None:
        """Load and prepare datasets.
        
        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            seed: Random seed for reproducible dataset splitting (default: None)
        """
        try:
            # Load dataset
            self.log_info(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)
            
            # Split if needed
            if 'test' not in dataset:
                self.log_info(f"Creating train/test split (test_size={test_size}, seed={seed})")
                dataset = dataset["train"].train_test_split(
                    test_size=test_size,
                    seed=seed
                )
            
            # Process splits
            self.datasets = {}
            for split in ['train', 'test']:
                split_size = len(dataset[split])
                self.log_info(f"\nProcessing {split} split:")
                self.log_info(f"Total examples: {split_size}")
                
                # Store last example for logging
                self._last_example = None
                self._current_split = split
                
                self.datasets[split] = dataset[split].map(
                    self._preprocess_example,
                    remove_columns=dataset[split].column_names,
                    batch_size=100,
                    keep_in_memory=True,
                    desc=f"Processing {split} split"
                )

                # Log format of last processed example
                if self._last_example and self.state_manager.is_main_process():
                    # Get the actual processed example
                    processed_example = self._preprocess_example({"audio": {"array": self._last_example["audio_array"], "sampling_rate": 16000}, "text": self._last_example["text"]})
                    
                    self.log_info("\nExample data format:")
                    self.log_info(f"Input audio shape: {self._last_example['audio_shape']}")
                    self.log_info(f"Input text: {self._last_example['text']}")
                    self.log_info(f"Raw features shape: {processed_example['input_features'].shape}")
                    self.log_info(f"Raw encoder attention mask shape: {processed_example['attention_mask'].shape}")
                    self.log_info(f"Raw decoder attention mask shape: {processed_example['decoder_attention_mask'].shape}")
                    self.log_info(f"Label length: {len(processed_example['labels'])}")
                    self.log_info("\nFinal batched shape would be:")
                    self.log_info(f"input_features: [batch_size, {processed_example['input_features'].shape[0]}, {processed_example['input_features'].shape[1]}]")
                    self.log_info(f"encoder_attention_mask: [batch_size, {processed_example['attention_mask'].shape[0]}]")
                    self.log_info(f"decoder_attention_mask: [batch_size, {processed_example['decoder_attention_mask'].shape[0]}]")
                    self.log_info(f"labels: [batch_size, {len(processed_example['labels'])}]")

                self.log_info(f"Completed processing {split} split")
                
            self.log_info("Dataset preparation completed")
            
        except Exception as e:
            self.log_error(f"Dataset preparation failed: {str(e)}")
            raise ExperimentError(f"Dataset preparation failed: {str(e)}") from e

    @property
    def train_dataset(self) -> Optional[Dataset]:
        """Get training dataset."""
        return self.datasets['train'] if self.datasets else None
    
    @property
    def test_dataset(self) -> Optional[Dataset]:
        """Get test dataset."""
        return self.datasets['test'] if self.datasets else None