import os
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
from huggingface_hub import login, HfFolder, HfApi
from tqdm import tqdm
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    EvalPrediction,
    GenerationConfig,
    EarlyStoppingCallback,
    TrainingArguments,
    TrainerCallback
)
from collections import defaultdict, deque
from pathlib import Path
import math
import collections
import sys
import subprocess

# Constants
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
READABLE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

print(f"Using device: {DEVICE}")

@dataclass
class RunTimestamp:
    """Centralized timestamp management for consistent file/folder naming"""
    created_at: datetime
    
    def get_formatted(self) -> str:
        """Get timestamp formatted for file/folder names"""
        return self.created_at.strftime(TIMESTAMP_FORMAT)
    
    def get_readable(self) -> str:
        """Get human readable timestamp"""
        return self.created_at.strftime(READABLE_TIMESTAMP_FORMAT)
    
    def get_elapsed(self, from_time: datetime = None) -> str:
        """Get elapsed time since timestamp or given time in HH:MM:SS format"""
        if from_time is None:
            from_time = datetime.now()
        elapsed = from_time - self.created_at
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def get_elapsed_detailed(self, from_time: datetime = None) -> str:
        """Get detailed elapsed time including days if applicable"""
        if from_time is None:
            from_time = datetime.now()
        elapsed = from_time - self.created_at
        days = elapsed.days
        hours, remainder = divmod(elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0 or days > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        
        return ", ".join(parts)

def create_run_timestamp() -> RunTimestamp:
    """Create a timestamp for the current run that will be used consistently throughout"""
    return RunTimestamp(created_at=datetime.now())

def create_model_card(config, dataset, baseline_wer, finetuned_wer, training_args):
    """Create a model card with training details and performance metrics."""
    import math
    
    # Safely get dataset sizes with fallbacks
    try:
        train_size = len(dataset.get('train', [])) if dataset else 0
        test_size = len(dataset.get('test', [])) if dataset else 0
    except Exception:
        train_size = 0
        test_size = 0
    
    # Safely get config attributes with fallbacks
    safe_config = {
        'model_name': getattr(config, 'model_name', 'unknown_model'),
        'dataset_name': getattr(config, 'dataset_name', 'unknown_dataset'),
        'batch_size': getattr(config, 'batch_size', 'Not specified'),
        'learning_rate': getattr(config, 'learning_rate', 'Not specified'),
        'warmup_steps': getattr(config, 'warmup_steps', 'Not specified'),
        'max_steps': getattr(config, 'max_steps', 'Not specified'),
        'weight_decay': getattr(config, 'weight_decay', 'Not specified'),
        'fp16': getattr(config, 'fp16', 'Not specified'),
        'gradient_checkpointing': getattr(config, 'gradient_checkpointing', 'Not specified')
    }
    
    # Sanitize model name for citation
    safe_model_name = ''.join(c if c.isalnum() else '_' for c in safe_config['model_name'].lower())
    
    # Get hub_model_id safely
    hub_model_id = getattr(training_args, 'hub_model_id', 'undefined_model_id')
    
    # Validate numeric metrics
    try:
        baseline_wer = float(baseline_wer)
        finetuned_wer = float(finetuned_wer)
        if math.isnan(baseline_wer) or math.isinf(baseline_wer):
            baseline_wer = 0.0
        if math.isnan(finetuned_wer) or math.isinf(finetuned_wer):
            finetuned_wer = 0.0
    except (TypeError, ValueError):
        baseline_wer = 0.0
        finetuned_wer = 0.0
    
    model_card = f"""# Whisper Fine-tuned Model

This model is a fine-tuned version of `{safe_config['model_name']}` on `{safe_config['dataset_name']}`.

## Model Description
- **Model Type:** Fine-tuned Whisper model for speech recognition
- **Language:** English
- **Task:** Automatic Speech Recognition
- **Domain:** Throat Microphone Speech Recognition

## Training Details
- **Base Model:** `{safe_config['model_name']}`
- **Dataset:** `{safe_config['dataset_name']}`
- **Training Examples:** {train_size}
- **Test Examples:** {test_size}
- **Training Steps:** {safe_config['max_steps']}

### Hyperparameters
- **Batch Size:** {safe_config['batch_size']}
- **Learning Rate:** {safe_config['learning_rate']}
- **Warmup Steps:** {safe_config['warmup_steps']}
- **Weight Decay:** {safe_config['weight_decay']}
- **FP16:** {safe_config['fp16']}
- **Gradient Checkpointing:** {safe_config['gradient_checkpointing']}

## Performance
- **Baseline WER:** {baseline_wer:.4f}
- **Fine-tuned WER:** {finetuned_wer:.4f}

## Usage

You can use this model as follows:

<pre><code class="language-python">
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load processor and model
processor = WhisperProcessor.from_pretrained("{hub_model_id}")
model = WhisperForConditionalGeneration.from_pretrained("{hub_model_id}")

# Example usage
inputs = processor("Audio input data", return_tensors="pt", sampling_rate=16000)
outputs = model.generate(inputs["input_features"])
transcription = processor.batch_decode(outputs, skip_special_tokens=True)
print(transcription)
</code></pre>

## Citation
If you use this model, please cite:

<pre><code class="language-bibtex">
@misc{{whisper_finetune_{safe_model_name}}},
  title={{{{Fine-tuned Whisper Model}}}},
  author={{{{Your Name or Team Name}}}},
  year={{{{2024}}}},
  howpublished={{https://huggingface.co/{hub_model_id}}}
</code></pre>

## Acknowledgments
Thanks to the Hugging Face team and the community for providing tools to fine-tune and deploy this model.
"""

    return model_card


class ExperimentLogger:
    """Simple logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, run_timestamp: RunTimestamp, print_metrics: bool = True, print_predictions: bool = True):
        self.experiment_name = experiment_name
        self.run_timestamp = run_timestamp
        
        # Create experiment directory under logs/
        self.experiment_dir = LOG_DIR / f"{experiment_name}_{run_timestamp.get_formatted()}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            "metadata": {
                "name": experiment_name,
                "start_time": run_timestamp.get_readable()
            },
            "timeline": []
        }
        
        # Output control flags
        self.print_metrics = print_metrics
        self.print_predictions = print_predictions
    
    def _get_elapsed_time(self):
        """Get elapsed time since experiment start"""
        return self.run_timestamp.get_elapsed()
    
    def _save_yaml(self):
        """Save current results to YAML file"""
        yaml_path = self.experiment_dir / "results.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(self.results, f, default_flow_style=False)
    
    def log(self, message: str):
        """Log message to terminal with timestamp"""
        elapsed = self._get_elapsed_time()
        print(f"[+{elapsed}] {message}")
        
        self.results["timeline"].append({
            "type": "message",
            "elapsed": elapsed,
            "content": message
        })
        self._save_yaml()
    
    def save_metric(self, name: str, value: float, print_to_terminal: bool = None):
        """Save a metric to the timeline"""
        # Validate numeric value
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            raise ValueError(f"Invalid metric value for {name}: {value}")
            
        elapsed = self._get_elapsed_time()
        
        # Determine whether to print based on instance setting or override
        should_print = print_to_terminal if print_to_terminal is not None else self.print_metrics
        if should_print:
            print(f"[+{elapsed}] Metric - {name}: {value:.4f}")
        
        self.results["timeline"].append({
            "type": "metric",
            "elapsed": elapsed,
            "name": name,
            "value": float(value)  # Ensure it's serializable
        })
        self._save_yaml()
    
    def save_prediction(self, reference: str, prediction: str, print_to_terminal: bool = None):
        """Save a prediction example to the timeline with WER metric.
        
        Args:
            reference: Reference text
            prediction: Model prediction
            print_to_terminal: Whether to print to terminal (overrides instance setting)
        """
        elapsed = self._get_elapsed_time()
        
        # Calculate WER for this prediction
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=[prediction], references=[reference])
        
        # Determine whether to print based on instance setting or override
        should_print = print_to_terminal if print_to_terminal is not None else self.print_predictions
        if should_print:
            print(f"[+{elapsed}] New prediction:")
            print(f"Reference: {reference}")
            print(f"Prediction: {prediction}")
            print(f"WER: {wer:.4f}")
        
        self.results["timeline"].append({
            "type": "prediction",
            "elapsed": elapsed,
            "reference": reference,
            "prediction": prediction,
            "wer": float(wer)  # Ensure it's serializable
        })
        self._save_yaml()

@dataclass
class ExperimentConfig:
    # Experiment metadata
    name: str
    date: str
    description: str
    
    # Model and dataset
    model_name: str
    dataset_name: str
    
    # Audio processing
    normalize_audio: bool
    
    # Training parameters
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_checkpointing: bool
    max_grad_norm: float
    fp16: bool
    lr_scheduler_type: str
    
    # Evaluation and saving
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    logging_steps: int
    save_total_limit: int
    push_to_hub: bool
    hub_model_id: str
    hub_strategy: str
    predict_with_generate: bool
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    report_to: List[str]
    
    # Regularization
    weight_decay: float
    dropout: float
    label_smoothing: float
    
    # Early stopping (simplified)
    early_stopping_enabled: bool
    early_stopping_patience: int
    early_stopping_threshold: float
    
    # Monitoring settings
    gradient_history_size: int
    significant_change_threshold: float
    log_top_n_gradients: int
    
    # Generation settings
    generation_language: str
    generation_task: str
    generation_max_length: int
    use_cache: bool
    
    # Dataset settings
    test_split_size: int
    test_split_seed: int
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle scientific notation for learning rate
        learning_rate = config['training']['learning_rate']
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate.replace('e-', 'e-'))
        
        # Handle max_grad_norm - convert to float or None
        max_grad_norm = config['training']['max_grad_norm']
        if isinstance(max_grad_norm, str) and max_grad_norm.lower() == 'none':
            max_grad_norm = None
        elif max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)
        
        return cls(
            name=config['experiment']['name'],
            date=config['experiment']['date'],
            description=config['experiment']['description'],
            model_name=config['environment']['base_model'],
            dataset_name=config['dataset']['source'],
            normalize_audio=config['dataset']['audio'].get('normalize', True),
            batch_size=int(config['training']['batch_size']),
            eval_batch_size=int(config['training']['eval_batch_size']),
            gradient_accumulation_steps=int(config['training']['gradient_accumulation_steps']),
            learning_rate=learning_rate,
            warmup_steps=int(config['training']['warmup_steps']),
            max_steps=int(config['training']['max_steps']),
            gradient_checkpointing=bool(config['training']['gradient_checkpointing']),
            max_grad_norm=max_grad_norm,  # Use the processed value
            fp16=bool(config['training']['fp16']),
            lr_scheduler_type=config['training']['lr_scheduler_type'],
            evaluation_strategy=config['training']['evaluation_strategy'],
            eval_steps=int(config['training']['eval_steps']),
            save_steps=int(config['training']['save_steps']),
            logging_steps=int(config['training']['logging_steps']),
            save_total_limit=int(config['training']['save_total_limit']),
            push_to_hub=bool(config['training']['push_to_hub']),
            hub_model_id=config['training']['hub_model_id'],
            hub_strategy=config['training']['hub_strategy'],
            predict_with_generate=bool(config['training']['predict_with_generate']),
            load_best_model_at_end=bool(config['training']['load_best_model_at_end']),
            metric_for_best_model=config['training']['metric_for_best_model'],
            greater_is_better=bool(config['training']['greater_is_better']),
            report_to=config['training']['report_to'],
            weight_decay=float(config['training']['regularization']['weight_decay']),
            dropout=float(config['training']['regularization']['dropout']),
            label_smoothing=float(config['training']['regularization']['label_smoothing']),
            early_stopping_enabled=bool(config['evaluation']['early_stopping']['enabled']),
            early_stopping_patience=int(config['evaluation']['early_stopping']['patience']),
            early_stopping_threshold=float(config['evaluation']['early_stopping'].get('threshold', 0.0001)),
            gradient_history_size=int(config['training']['monitoring']['gradient_history_size']),
            significant_change_threshold=float(config['training']['monitoring']['significant_change_threshold']),
            log_top_n_gradients=int(config['training']['monitoring']['log_top_n_gradients']),
            generation_language=config['training']['generation']['language'],
            generation_task=config['training']['generation']['task'],
            generation_max_length=int(config['training']['generation']['max_length']),
            use_cache=bool(config['training']['generation']['use_cache']),
            test_split_size=int(config['dataset']['test_split']),
            test_split_seed=config.get('dataset', {}).get('seed', 42),
        )

def prepare_dataset(batch, idx):
    try:
        # Load and resample audio
        audio = batch["audio"]
        
        # Process audio - ensure we're using 16kHz which is what Whisper expects
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            normalize=config.normalize_audio  # Keep normalization config
        ).input_features[0]  # Remove first dim as we'll batch later
        
        # Store features
        batch["input_features"] = input_features
        
        # Process text (labels)
        batch["labels"] = processor.tokenizer(batch["text"], truncation=True).input_ids
        
        return batch
    except Exception as e:
        logger.log(f"Error processing example {idx}: {str(e)}")
        raise e

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = None  # Class property for decoder start token
    has_logged_example: bool = False    # Track if we've logged an example

    def __init__(self, processor: Any):
        self.processor = processor
        # Get decoder start token ID once during initialization
        self.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        #labels = labels_batch["input_ids"].masked_fill(labels_batch.input_ids == self.processor.tokenizer.pad_token_id, -100)  # Where token is pad_token



        # Log a single example if we haven't already
        if not self.has_logged_example:
            # Get the first example from the batch
            example_ids = labels.tolist()[0]
            example_mask = labels_batch.attention_mask[0].tolist()
            
            # Decode the tokens for better readability
            decoded_tokens = [self.processor.tokenizer.decode([token]) for token in example_ids]
            
            print("\nExample Text Sequence:")
            print("-" * 80)
            print("Position | Token | Token Text | Attention Mask | Loss Mask")
            print("-" * 80)
            for pos, (token_id, attn_mask, decoded) in enumerate(zip(example_ids, example_mask, decoded_tokens)):
                loss_mask = "Yes" if token_id != -100 else "No"
                attn = "Yes" if attn_mask == 1 else "No"
                print(f"{pos:8d} | {token_id:5d} | {decoded:10s} | {attn:13s} | {loss_mask:9s}")
            print("-" * 80)
            print(f"Audio normalization: {config.normalize_audio}")
            print(f"Decoder start token ID: {self.decoder_start_token_id}")
            print("-" * 80)
            
            self.has_logged_example = True

        # Set the labels in the batch
        batch["labels"] = labels
        
        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def evaluate_model(model, dataset, processor, split="test"):
    logger.log(f"Evaluating model on {split} split...")
    model.eval()
    predictions = []
    references = []
    
    for i, item in enumerate(tqdm(dataset[split])):
        # Process audio with attention mask
        processor_kwargs = {
            "sampling_rate": item["audio"]["sampling_rate"],
            "return_tensors": "pt",
            "normalize": config.normalize_audio,
        }
            
        inputs = processor(
            item["audio"]["array"],
            **processor_kwargs
        ).to(DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            generate_kwargs = {
                "input_features": inputs.input_features,
                "language": config.generation_language,
                "task": config.generation_task,
                "use_cache": config.use_cache
            }
                
            predicted_ids = model.generate(
                **generate_kwargs
            )[0]
        
        # Decode prediction
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        predictions.append(transcription)
        references.append(item["text"])
        
    # Compute metrics
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Print some example predictions
    logger.log("Example predictions:")
    for pred, ref in list(zip(predictions, references))[:3]:
        logger.save_prediction(ref, pred)
    
    logger.save_metric("wer", wer)
    return wer



class TranscriptionLoggingCallback(TrainerCallback):
    """Callback to log transcription examples and training metrics during training.
    
    Logs sample transcriptions during evaluation steps and training metrics from the trainer.
    Uses batch processing for efficient transcription generation.
    
    Args:
        eval_dataset: Dataset to sample examples from
        processor: Whisper processor for transcription
        config: Experiment configuration containing test_split_seed
        experiment_logger: Logger for tracking metrics and examples
        num_examples: Number of examples to log per evaluation (default: 5)
        max_logged_steps: Size of logging history buffer (default: 1000)
    """
    
    def __init__(self, eval_dataset, processor, config, experiment_logger, 
                 num_examples=5, max_logged_steps=1000):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.config = config
        self.experiment_logger = experiment_logger
        self.tb_writer = None
        self.run_timestamp = experiment_logger.run_timestamp  # Get timestamp from logger
        
        # Set random seed for reproducible example selection
        np.random.seed(config.test_split_seed)
        
        # Select fixed examples and cache them
        self.num_examples = min(num_examples, len(eval_dataset))
        indices = np.random.choice(len(eval_dataset), self.num_examples, replace=False)
        self.examples = [self.eval_dataset[int(idx)] for idx in indices]
        
        # Log selected indices for reproducibility
        self.experiment_logger.log(f"Selected example indices for tracking: {indices.tolist()}")
        
        # Reset random seed to avoid affecting other random operations
        np.random.seed(None)
        
        # Pre-process audio inputs
        self.audio_inputs = self.processor(
            [ex["audio"]["array"] for ex in self.examples],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Cache reference texts
        self.references = [ex["text"] for ex in self.examples]
        
        # Track logged steps to prevent duplicates
        self.logged_steps = collections.deque(maxlen=max_logged_steps)

    def on_init_end(self, args, state, control, **kwargs):
        """Initialize TensorBoard writer if enabled."""
        if "tensorboard" in args.report_to:
            # Use the logs/runs directory for tensorboard
            tensorboard_dir = LOG_DIR / "runs" / f"{self.config.name}_{self.run_timestamp.get_formatted()}"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
            self.experiment_logger.log(f"Initialized TensorBoard writer in {tensorboard_dir}")
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Log transcription examples during evaluation using batch processing."""
        if state.global_step in self.logged_steps:
            return
        
        self.experiment_logger.log(f"\nTranscription Examples (Step {state.global_step}):")
        self.experiment_logger.log("-" * 50)
        
        try:
            model.eval()
            
            # Move inputs to model device
            inputs = self.audio_inputs.to(model.device)
            
            # Generate all predictions in one batch
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features=inputs.input_features,
                    language=self.config.generation_language,
                    task=self.config.generation_task,
                    max_length=self.config.generation_max_length,
                    use_cache=self.config.use_cache
                )
            
            # Decode all predictions at once
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            # Compute WER for each example
            wer_metric = evaluate.load("wer")
            wers = [
                wer_metric.compute(predictions=[transcription], references=[reference])
                for transcription, reference in zip(transcriptions, self.references)
            ]            
            # Log all examples
            for idx, (reference, transcription, wer) in enumerate(zip(self.references, transcriptions, wers)):
                # Log to experiment logger
                self.experiment_logger.save_prediction(reference, transcription)
                
                # Log to TensorBoard if available
                if self.tb_writer:
                    self.tb_writer.add_text(
                        f'Transcriptions/Example_{idx}',
                        f'**Reference**:\n```\n{reference}\n```\n\n'
                        f'**Prediction**:\n```\n{transcription}\n```\n\n'
                        f'**WER**: {wer:.4f}',
                        global_step=state.global_step
                    )
            
            self.logged_steps.append(state.global_step)
            
        except Exception as e:
            self.experiment_logger.log(f"Failed to log transcriptions: {str(e)}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics from the trainer."""
        if not logs:
            return
        
        # Format step information
        step_info = f"Step {state.global_step}"
        if 'epoch' in logs:
            step_info += f" (Epoch {logs['epoch']:.2f})"
        
        self.experiment_logger.log(f"\nTraining Metrics - {step_info}:")
        
        # Log each metric
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                try:
                    self.experiment_logger.save_metric(
                        name=f"train_{key}",
                        value=float(value),
                        print_to_terminal=False
                    )
                except ValueError as e:
                    self.experiment_logger.log(f"Failed to log metric {key}: {str(e)}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup TensorBoard writer."""
        if self.tb_writer:
            try:
                self.tb_writer.close()
                self.experiment_logger.log("Closed TensorBoard writer")
            except Exception as e:
                self.experiment_logger.log(f"Failed to close TensorBoard writer: {str(e)}")


if __name__ == "__main__":

    # Get the directory where the script is located
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir / 'config.yaml'
    
    # Check Git LFS installation early
    print("\nChecking Git LFS installation...")
    try:
        # Check git-lfs version
        lfs_version = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
        if lfs_version.returncode != 0:
            print("Error: git-lfs is not installed. This is required for model upload.")
            print("Please install it first:")
            print("  Ubuntu: sudo apt-get install git-lfs")
            print("  macOS: brew install git-lfs")
            print("  Windows: https://git-lfs.github.com")
            sys.exit(1)
            
        # Check git-lfs is initialized
        lfs_status = subprocess.run(['git', 'lfs', 'status'], capture_output=True, text=True)
        if lfs_status.returncode != 0:
            print("Git LFS is installed but not initialized.")
            print("Please run: git lfs install")
            sys.exit(1)
            
        print(f"Git LFS is properly installed and configured:")
        print(f"Version: {lfs_version.stdout.splitlines()[0]}")
    except FileNotFoundError:
        print("Error: git command not found. Please install Git first.")
        sys.exit(1)
    
    # Check HuggingFace token early
    print("\nValidating HuggingFace token...")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not found")
        print("Please set it with: export HF_TOKEN=your_token")
        print("Or run: huggingface-cli login")
        sys.exit(1)

    try:
        # Try to use the token with HfApi
        api = HfApi(token=token)
        # Try to get user info to validate token
        api.whoami()
        print("HuggingFace token is valid and working")
    except Exception as e:
        print(f"Error: HuggingFace token validation failed: {str(e)}")
        print("Please check your token and ensure you're properly logged in")
        sys.exit(1)
    
    # Load configuration
    config = ExperimentConfig.from_yaml(default_config)
    
    # Create a single timestamp for the entire run
    run_timestamp = create_run_timestamp()
    
    # Initialize logger with the run timestamp
    logger = ExperimentLogger(config.name, run_timestamp)
    
    # Create output directory for this run
    run_output_dir = OUTPUT_DIR / f"{config.name}_{run_timestamp.get_formatted()}"
    
    # Validate directory structure
    logger.log("\nValidating directory structure...")
    
    # Ensure all required directories exist
    required_dirs = [
        LOG_DIR,
        LOG_DIR / "runs",  # Add runs directory under logs
        OUTPUT_DIR,
        logger.experiment_dir,
        run_output_dir
    ]
    
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        logger.log(f"Verified directory: {directory}")
    
    logger.log("Directory structure validation complete\n")
    
    # Log experiment initialization
    logger.log("=" * 50)
    logger.log("EXPERIMENT INITIALIZATION")
    logger.log("=" * 50)
    logger.log(f"Name: {config.name}")
    logger.log(f"Description: {config.description}")
    logger.log(f"Date: {run_timestamp.get_readable()}")
    
    # Log configuration
    logger.log("\nCONFIGURATION:")
    logger.log("-" * 20)
    logger.log("Training Parameters:")
    logger.log(f"  Batch Size: {config.batch_size}")
    logger.log(f"  Learning Rate: {config.learning_rate}")
    logger.log(f"  Max Steps: {config.max_steps}")
    logger.log(f"  Warmup Steps: {config.warmup_steps}")
    logger.log(f"  FP16: {config.fp16}")
    logger.log(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
    logger.log(f"  Max Gradient Norm: {config.max_grad_norm}")
    
    # Log environment
    logger.log("\nENVIRONMENT:")
    logger.log("-" * 20)
    logger.log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Log model information
    logger.log("\nMODEL:")
    logger.log("-" * 20)
    logger.log(f"Base Model: {config.model_name}")
    
    # Login to HuggingFace
    login(token)  # We already validated the token at script start
    
    # Dataset loading and logging
    logger.log("\nDATASET:")
    logger.log("-" * 20)
    logger.log(f"Source: {config.dataset_name}")
    
    dataset = load_dataset(config.dataset_name, download_mode="force_redownload") #make this a script argument
    total_examples = len(dataset["train"])
    logger.log(f"Total Examples: {total_examples}")
    
    # Create and log train/test split
    dataset = dataset["train"].train_test_split(
        test_size=config.test_split_size, 
        seed=config.test_split_seed
    )
    logger.log("\nData Split:")
    logger.log(f"  Training Examples: {len(dataset['train'])}")
    logger.log(f"  Testing Examples: {len(dataset['test'])}")
    logger.log(f"  Test Split Ratio: {len(dataset['test'])/total_examples:.2%}")
    
    # Audio processing settings
    logger.log("\nAudio Processing:")
    logger.log(f"  Normalization: {config.normalize_audio}")
    
    # After other logging sections, before "=" * 50
    
    # Evaluation Settings
    logger.log("\nEvaluation Settings:")
    logger.log("-" * 20)
    logger.log(f"  Strategy: {config.evaluation_strategy}")
    logger.log(f"  Eval Steps: {config.eval_steps}")
    logger.log(f"  Save Steps: {config.save_steps}")
    logger.log(f"  Logging Steps: {config.logging_steps}")
    logger.log(f"  Load Best Model: {config.load_best_model_at_end}")
    logger.log(f"  Metric: {config.metric_for_best_model}")
    logger.log(f"  Greater is Better: {config.greater_is_better}")
    logger.log(f"  Save Total Limit: {config.save_total_limit}")
    

    
    logger.log("=" * 50)
    
    logger.log("\n" + "=" * 50)
    logger.log("STARTING EXPERIMENT")
    logger.log("=" * 50 + "\n")
    
    # Load model and processor
    logger.log(f"Loading model and processor: {config.model_name}")
    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name).to(DEVICE)
    
    # Configure generation settings
    model.generation_config.language = config.generation_language
    model.generation_config.task = config.generation_task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.use_cache = config.use_cache
    
    logger.log("Model and processor loaded successfully\n")
    
    # Evaluate baseline performance
    logger.log("Evaluating baseline model...")
    baseline_wer = evaluate_model(model, dataset, processor)
    logger.save_metric("baseline_wer", baseline_wer)
    
    # Prepare dataset
    logger.log("\nPreparing dataset for training...")
    
    # Process train split
    logger.log("Processing train split...")
    processed_train = []
    for idx, example in enumerate(tqdm(dataset["train"])):
        processed_train.append(prepare_dataset(example, idx))
    logger.log(f"Processed {len(processed_train)} training examples")
    
    # Process test split
    logger.log("Processing test split...")
    processed_test = []
    for idx, example in enumerate(tqdm(dataset["test"])):
        processed_test.append(prepare_dataset(example, idx))
    logger.log(f"Processed {len(processed_test)} test examples")
    
    # Create new dataset
    from datasets import Dataset
    processed_dataset = {}
    processed_dataset["train"] = Dataset.from_list(processed_train)
    processed_dataset["test"] = Dataset.from_list(processed_test)
    dataset = processed_dataset
    
    logger.log("Dataset preparation complete")
    
    # Prepare training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        remove_unused_columns=False,
        generation_max_length=config.generation_max_length,
        predict_with_generate=config.predict_with_generate,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,  # Keep this False initially
        lr_scheduler_type=config.lr_scheduler_type,
        run_name=config.name  # Add explicit run_name to avoid timestamp appending
    )
    
    # Log training argument details after they're created
    logger.log("\nTraining Configuration:")
    logger.log("=" * 50)
    
    # Core Training Parameters
    logger.log("\nCore Training Parameters:")
    logger.log("-" * 20)
    logger.log(f"  Learning Rate: {training_args.learning_rate}")
    logger.log(f"  LR Scheduler: {training_args.lr_scheduler_type}")
    logger.log(f"  Max Steps: {training_args.max_steps}")
    logger.log(f"  Warmup Steps: {training_args.warmup_steps}")
    logger.log(f"  Warmup Ratio: {training_args.warmup_ratio:.2%}")
    logger.log(f"  Optimizer: AdamW")  # Currently hardcoded in Seq2SeqTrainer


    # Generation Settings
    logger.log("\nGeneration Settings:")
    logger.log("-" * 20)
    logger.log(f"  Generation Max Length: {training_args.generation_max_length}")
    logger.log(f"  Predict with Generate: {training_args.predict_with_generate}")
    logger.log(f"  Language: {config.generation_language}")
    logger.log(f"  Task: {config.generation_task}")
    logger.log(f"  Use Cache: {config.use_cache}")
    
    # Batch Configuration
    logger.log("\nBatch Configuration:")
    logger.log("-" * 20)
    logger.log(f"  Per Device Train Batch Size: {training_args.per_device_train_batch_size}")
    logger.log(f"  Per Device Eval Batch Size: {training_args.per_device_eval_batch_size}")
    logger.log(f"  Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
    effective_batch_size = (training_args.per_device_train_batch_size * 
                         training_args.gradient_accumulation_steps * 
                         torch.cuda.device_count())
    logger.log(f"  Effective Batch Size: {effective_batch_size}")
    
    # Regularization
    logger.log("\nRegularization:")
    logger.log("-" * 20)
    logger.log(f"  Weight Decay: {training_args.weight_decay}")
    logger.log(f"  Label Smoothing: {training_args.label_smoothing_factor}")
    logger.log(f"  Dropout: {config.dropout}")
    logger.log(f"  Max Gradient Norm: {training_args.max_grad_norm}")
    
    # Memory Optimization
    logger.log("\nMemory Optimization:")
    logger.log("-" * 20)
    logger.log(f"  Gradient Checkpointing: {training_args.gradient_checkpointing}")
    logger.log(f"  FP16 Training: {training_args.fp16}")
    logger.log(f"  FP16 Opt Level: {training_args.fp16_opt_level}")
    logger.log(f"  FP16 Backend: {training_args.fp16_backend}")
    
    # Evaluation & Saving Configuration
    logger.log("\nEvaluation & Saving Configuration:")
    logger.log("-" * 20)
    logger.log(f"  Evaluation Strategy: {training_args.evaluation_strategy}")
    logger.log(f"  Eval Steps: {training_args.eval_steps}")
    logger.log(f"  Save Steps: {training_args.save_steps}")
    logger.log(f"  Logging Steps: {training_args.logging_steps}")
    logger.log(f"  Save Total Limit: {training_args.save_total_limit}")
    logger.log(f"  Load Best Model at End: {training_args.load_best_model_at_end}")
    logger.log(f"  Metric for Best Model: {training_args.metric_for_best_model}")
    logger.log(f"  Greater is Better: {training_args.greater_is_better}")
    
    # Early Stopping Configuration
    logger.log("\nEarly Stopping Configuration:")
    logger.log("-" * 20)
    logger.log(f"  Enabled: {config.early_stopping_enabled}")
    if config.early_stopping_enabled:
        logger.log(f"  Patience: {config.early_stopping_patience}")
        logger.log(f"  Threshold: {config.early_stopping_threshold}")
    
    # Output & Logging
    logger.log("\nOutput & Logging Configuration:")
    logger.log("-" * 20)
    logger.log(f"  Output Directory: {training_args.output_dir}")
    logger.log(f"  Logging Directory: {training_args.logging_dir}")
    logger.log(f"  Report To: {training_args.report_to}")
    logger.log(f"  Push to Hub: {training_args.push_to_hub}")
    if training_args.push_to_hub:
        logger.log(f"  Hub Model ID: {training_args.hub_model_id}")
        logger.log(f"  Hub Strategy: {training_args.hub_strategy}")
    
    # Monitoring Configuration
    logger.log("\nMonitoring Configuration:")
    logger.log("-" * 20)
    logger.log(f"  Gradient History Size: {config.gradient_history_size}")
    logger.log(f"  Significant Change Threshold: {config.significant_change_threshold}")
    logger.log(f"  Log Top N Gradients: {config.log_top_n_gradients}")
    
    logger.log("=" * 50)
    
    # Add early stopping if enabled
    callbacks = [TranscriptionLoggingCallback(dataset["test"], processor, config, logger)]
    if config.early_stopping_enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        )
    
    # Initialize trainer with default Seq2SeqTrainer and our new callback
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks
    )
    


    # Fine-tune the model
    logger.log("\nStarting fine-tuning...")
    trainer.train()
    
    # Save the final model (which is the best model due to load_best_model_at_end=True)
    trainer.save_model()  # This will save to output_dir from training_args
    processor.save_pretrained(run_output_dir)  # Save processor alongside model
    
    # Evaluate fine-tuned model
    logger.log("\nEvaluating fine-tuned model...")
    finetuned_wer = evaluate_model(model, dataset, processor)
    
    # Save results
    logger.log("\nSaving WER to txt")
    results_file = LOG_DIR / f"{config.name}_{run_timestamp.get_formatted()}_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Baseline WER: {baseline_wer:.4f}\n")
        f.write(f"Fine-tuned WER: {finetuned_wer:.4f}\n")
        f.write(f"Total Training Time: {run_timestamp.get_elapsed_detailed()}\n")
    
    logger.log("\nExperiment complete!")
    logger.log(f"Baseline WER: {baseline_wer:.4f}")
    logger.log(f"Fine-tuned WER: {finetuned_wer:.4f}")
    logger.log(f"Improvement: {(baseline_wer - finetuned_wer):.4f} absolute / {((baseline_wer - finetuned_wer)/baseline_wer * 100):.1f}% relative")
    logger.log(f"Total Training Time: {run_timestamp.get_elapsed_detailed()}")
    logger.log("=" * 50)
    
    # After training and evaluation is complete
    if config.push_to_hub:
        # Validate hub_model_id
        if not config.hub_model_id:
            logger.log("Error: No hub_model_id specified in configuration")
            logger.log("Skipping upload to Hugging Face Hub")
            sys.exit(1)
        
        logger.log("\nTraining completed. Would you like to upload the model to Hugging Face Hub?")
        logger.log(f"  Model Performance:")
        logger.log(f"  - Baseline WER: {baseline_wer:.4f}")
        logger.log(f"  - Fine-tuned WER: {finetuned_wer:.4f}")
        logger.log(f"  Repository: {config.hub_model_id}")
        
        # More robust input handling
        while True:
            try:
                response = input("[y/N]: ").lower().strip()
                if response in ['y', 'n', '']:
                    break
                logger.log("Please enter 'y' for yes or 'n' (or just press Enter) for no.")
            except EOFError:
                logger.log("Input was interrupted. Please try again.")
            except KeyboardInterrupt:
                logger.log("\nOperation cancelled by user.")
                sys.exit(0)
        
        if response == 'y':
            try:
                logger.log("\nPreparing for upload to Hugging Face Hub...")
                
                # Create the repository if it doesn't exist
                try:
                    api.create_repo(
                        repo_id=config.hub_model_id,
                        private=True,
                        exist_ok=True,
                        token=token
                    )
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.log(f"Repository {config.hub_model_id} already exists, continuing with upload...")
                    else:
                        logger.log(f"Warning: Repository creation issue: {str(e)}")
                        logger.log("Will attempt to upload anyway...")
                
                # Create and save model card
                model_card = create_model_card(
                    config=config,
                    dataset=dataset,
                    baseline_wer=baseline_wer,
                    finetuned_wer=finetuned_wer,
                    training_args=training_args
                )
                
                with open(run_output_dir / "README.md", "w") as f:
                    f.write(model_card)
                
                # Configure trainer for hub upload
                trainer.args.push_to_hub = True
                trainer.args.hub_model_id = config.hub_model_id
                trainer.args.hub_strategy = config.hub_strategy
                
                # Push to hub using the trainer
                logger.log("Uploading model to Hugging Face Hub...")
                try:
                    trainer.push_to_hub(
                        commit_message=f"Training completed - WER: {finetuned_wer:.4f}",
                        blocking=True
                    )
                    logger.log(f"Model successfully uploaded to: {config.hub_model_id}")
                    logger.log(f"View your model at: https://huggingface.co/{config.hub_model_id}")
                except Exception as upload_error:
                    logger.log(f"Error during model upload: {str(upload_error)}")
                    logger.log("Attempting to diagnose the issue...")
                    
                    # Check common issues
                    try:
                        api.model_info(config.hub_model_id)
                        logger.log("Repository exists and is accessible.")
                    except Exception as e:
                        logger.log(f"Repository access check failed: {str(e)}")
                    
                    logger.log("Model saved locally but upload failed")
                    sys.exit(1)
                
            except Exception as e:
                logger.log(f"Error during upload preparation: {str(e)}")
                logger.log("Model saved locally but upload failed")
                sys.exit(1)
        else:
            logger.log("Upload cancelled. Model saved locally only")


