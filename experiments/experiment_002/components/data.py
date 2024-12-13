"""Data processing and preparation components for Whisper fine-tuning."""

from typing import Dict, Any, Optional, List, Union, cast, Protocol, runtime_checkable
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, field  # Added 'field' import
from datasets import Dataset, load_dataset, DatasetDict  # Updated import
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    BatchEncoding,
    BatchFeature
)
from transformers.tokenization_utils_fast import EncodingFast

from .logger import LoggerManager
from .state import StateManager

@runtime_checkable
class TypedWhisperProcessor(Protocol):
    """Protocol for type-annotated WhisperProcessor.

    This matches the actual WhisperProcessor implementation from Hugging Face.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/processing_whisper.py
    """
    def __init__(self, feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer) -> None:
        ...

    def __call__(
        self, 
        audio: Union[np.ndarray, List[float], List[List[float]]], 
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs: Any
    ) -> BatchFeature:
        """Process audio inputs."""
        ...

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = False,
        **kwargs: Any
    ) -> List[str]:
        """Decode token sequences to text."""
        ...

    def get_decoder_prompt_ids(
        self,
        language: Optional[str] = None,
        task: Optional[str] = None,
        no_timestamps: bool = False
    ) -> List[int]:
        """Get decoder prompt IDs."""
        ...

    @property
    def feature_extractor(self) -> WhisperFeatureExtractor:
        """Get feature extractor."""
        ...

    @property
    def tokenizer(self) -> WhisperTokenizer:
        """Get tokenizer."""
        ...

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper that handles proper decoder inputs and padding."""
    processor: TypedWhisperProcessor
    logger_manager: Optional[LoggerManager] = None
    state_manager: Optional[StateManager] = None  # To determine the main process
    batch_count: int = field(default=0, init=False)  # Initialize batch count with 'field'
    log_interval: int = field(default=500, init=False)  # Log every 500 batches
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features or not isinstance(features[0], dict):
            raise ValueError(f"Expected a non-empty list of dicts, got {type(features)}")
        
        # Prepare audio inputs - ensure each feature has shape (n_mels, time)
        input_features = []
        for feature in features:
            feat = feature["input_features"]
            # Handle numpy arrays
            if isinstance(feat, np.ndarray):
                if feat.ndim == 3 and feat.shape[0] == 1:
                    feat = feat.squeeze(0)
            # Handle torch tensors
            elif isinstance(feat, torch.Tensor):
                if feat.ndim == 3 and feat.shape[0] == 1:
                    feat = feat.squeeze(0)
            input_features.append({"input_features": feat})
        
        # Pad the batch and get attention_mask
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt"
        )
        
        # Reshape input features from [batch_size, 1, n_mels, time] to [batch_size, n_mels, time]
        if batch['input_features'].ndim == 4:
            batch['input_features'] = batch['input_features'].squeeze(1)
        
        # Increment batch_count and determine if logging should occur
        self.batch_count += 1
        should_log = (
            self.batch_count % self.log_interval == 0 and
            self.logger_manager is not None and
            self.state_manager is not None and
            self.state_manager.is_main_process()
        )

        if should_log:
            # Cast to non-optional types to satisfy Pylance
            logger = cast(LoggerManager, self.logger_manager)
            logger.log_info(f"[Batch {self.batch_count}] Input features shape after padding and reshaping: {batch['input_features'].shape}")
            logger.log_info(f"[Batch {self.batch_count}] Attention mask shape: {batch['attention_mask'].shape}")
        
        # Get the prompt length for masking
        prompt_text = ""  # No prompt in this case
        prompt_tokens: List[int] = self.processor.tokenizer(prompt_text).input_ids
        prompt_len = len(prompt_tokens)
        
        # Prepare text/label inputs
        text_features = [{"input_ids": f["labels"]} for f in features]
        
        # First, create decoder_input_ids
        decoder_batch: BatchEncoding = self.processor.tokenizer.pad(
            text_features,
            padding=True,
            return_tensors="pt"
        )

        # Now decoder_batch is a BatchEncoding containing tensors
        decoder_input_ids: torch.Tensor = cast(torch.Tensor, decoder_batch["input_ids"])

        if should_log:
            logger.log_info(f"[Batch {self.batch_count}] Decoder input_ids after padding: {decoder_input_ids.shape}")

        batch["decoder_input_ids"] = decoder_input_ids

        # Pad labels
        labels_batch: BatchEncoding = self.processor.tokenizer.pad(
            text_features,
            padding=True,
            return_tensors="pt"
        )

        labels_input_ids: torch.Tensor = cast(torch.Tensor, labels_batch["input_ids"])

        if should_log:
            logger.log_info(f"[Batch {self.batch_count}] Labels after padding: {labels_input_ids.shape}")

        # Replace padding with -100
        labels_padded = labels_input_ids.masked_fill(
            labels_input_ids == self.processor.tokenizer.pad_token_id,
            -100
        )
        
        # Mask out prompt tokens in labels
        if prompt_len > 0:
            labels_padded[:, :prompt_len] = -100
        
        if should_log:
            logger.log_info(f"[Batch {self.batch_count}] Labels after masking prompt tokens: {labels_padded.shape}")
        
        batch["labels"] = labels_padded

        # **Ensure attention_mask is present in the batch**
        # Whisper models primarily use 'input_features', but if 'attention_mask' is required, ensure it's included
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones(batch['input_features'].shape[:-1], dtype=torch.long)
            if should_log:
                logger.log_info(f"[Batch {self.batch_count}] Attention mask manually set: {batch['attention_mask'].shape}")
        
        return cast(Dict[str, torch.Tensor], batch)

@dataclass
class DataManager:
    """Manages dataset operations for Whisper fine-tuning."""
    processor: TypedWhisperProcessor
    dataset_name: str
    logger_manager: LoggerManager
    state_manager: StateManager
    config: Dict[str, Any]
    datasets: Optional[Dict[str, Dataset]] = field(default=None, init=False)
    collator: DataCollatorSpeechSeq2SeqWithPadding = field(init=False)
    example_count: int = field(default=0, init=False)
    log_interval: int = field(default=500, init=False)

    def __post_init__(self):
        self.collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            logger_manager=self.logger_manager,
            state_manager=self.state_manager
        )

    def prepare_datasets(self, test_size: float = 0.2, seed: int = 42) -> None:
        """Load and prepare datasets for training."""
        if self.logger_manager and self.state_manager.is_main_process():
            self.logger_manager.log_info(f"Loading dataset: {self.dataset_name}")
        
        # Load dataset (using Hugging Face Datasets as an example)
        raw_dataset = load_dataset(self.dataset_name)
        dataset = cast(DatasetDict, raw_dataset)  # Assuming it's a DatasetDict
        
        if self.logger_manager and self.state_manager.is_main_process():
            self.logger_manager.log_info(f"Dataset split before processing: {dataset}")
        
        # Split if needed
        if isinstance(dataset, DatasetDict):
            if 'test' not in dataset:
                if self.logger_manager and self.state_manager.is_main_process():
                    self.logger_manager.log_info("Splitting train into train and test splits")
                train_dataset = cast(Dataset, dataset["train"])
                dataset = train_dataset.train_test_split(test_size=test_size, seed=seed)
            else:
                if self.logger_manager and self.state_manager.is_main_process():
                    self.logger_manager.log_info("Splitting dataset into train and test splits")
                train_dataset = cast(Dataset, dataset["train"])
                dataset = train_dataset.train_test_split(test_size=test_size, seed=seed)
        
        if self.logger_manager and self.state_manager.is_main_process():
            self.logger_manager.log_info(f"Dataset split after processing: {dataset}")
        
        self.datasets = {}
        if self.logger_manager and self.state_manager.is_main_process():
            self.logger_manager.log_info("Processing splits: ['train', 'test']")
        for split in ['train', 'test']:
            split_dataset = cast(Dataset, dataset[split])
            self.datasets[split] = split_dataset.map(
                self._preprocess_example,
                remove_columns=split_dataset.column_names,
                desc=f"Processing {split} split"
            )
        if self.logger_manager and self.state_manager.is_main_process():
            self.logger_manager.log_info("Finished preparing datasets")

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Union[np.ndarray, List[int]]]:
        """Process a single example."""
        self.example_count += 1
        should_log = (
            self.example_count % self.log_interval == 0 and
            self.logger_manager is not None and
            self.state_manager is not None and
            self.state_manager.is_main_process()
        )
        
        if should_log:
            logger = cast(LoggerManager, self.logger_manager)
            logger.log_info(f"[Example {self.example_count}] Preprocessing a new example")
        
        # Process audio - keep as numpy array until collation
        batch_features = self.processor(
            example["audio"]["array"],
            sampling_rate=example["audio"]["sampling_rate"],
            return_tensors=None  # Don't return tensors yet
        )
        input_features = cast(np.ndarray, batch_features.input_features)
        
        # Ensure correct shape (n_mels, time) for single example
        if input_features.ndim == 3 and input_features.shape[0] == 1:
            input_features = input_features.squeeze(0)  # Remove batch dimension if present
            
        if should_log:
            logger.log_info(f"[Example {self.example_count}] Input features shape after processing: {input_features.shape}")
        
        # Process text - keep as list until collation
        encoding = self.processor.tokenizer(
            example["text"],
            truncation=True,
            max_length=448
            # Removed 'language' and 'mode' arguments
        )
        labels = encoding.input_ids  # type: ignore
        
        if should_log:
            logger.log_info(f"[Example {self.example_count}] Labels length after processing: {len(labels)}")
        
        return {
            "input_features": input_features,
            "labels": labels
        }
    @property
    def train_dataset(self) -> Optional[Dataset]:
        """Get training dataset."""
        return self.datasets.get('train') if self.datasets else None
        
    @property
    def test_dataset(self) -> Optional[Dataset]:
        """Get test dataset."""
        return self.datasets.get('test') if self.datasets else None
