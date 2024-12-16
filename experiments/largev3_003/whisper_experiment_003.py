import os
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
from huggingface_hub import login, HfFolder, Repository
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

# Constants
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

print(f"Using device: {DEVICE}")


def create_model_card(config, dataset, baseline_wer, finetuned_wer, training_args):
    """Create a model card with training details and performance metrics."""
    model_card = f"""# Whisper Fine-tuned Model

This model is a fine-tuned version of `{config.model_name}` on `{config.dataset_name}`.

## Model Description
- **Model Type:** Fine-tuned Whisper model for speech recognition
- **Language:** English
- **Task:** Automatic Speech Recognition
- **Domain:** Throat Microphone Speech Recognition

## Training Details
- **Base Model:** `{config.model_name}`
- **Dataset:** `{config.dataset_name}`
- **Training Examples:** {len(dataset['train'])}
- **Test Examples:** {len(dataset['test'])}
- **Training Steps:** {config.max_steps}

### Hyperparameters
- **Batch Size:** {config.batch_size}
- **Learning Rate:** {config.learning_rate}
- **Warmup Steps:** {config.warmup_steps}
- **Weight Decay:** {config.weight_decay}
- **FP16:** {config.fp16}
- **Gradient Checkpointing:** {config.gradient_checkpointing}

## Performance
- **Baseline WER:** {baseline_wer:.4f}
- **Fine-tuned WER:** {finetuned_wer:.4f}

## Usage

You can use this model as follows:

<pre><code class="language-python">
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load processor and model
processor = WhisperProcessor.from_pretrained("{training_args.hub_model_id}")
model = WhisperForConditionalGeneration.from_pretrained("{training_args.hub_model_id}")

# Example usage
inputs = processor("Audio input data", return_tensors="pt", sampling_rate=16000)
outputs = model.generate(inputs["input_features"])
transcription = processor.batch_decode(outputs, skip_special_tokens=True)
print(transcription)
</code></pre>

## Citation
If you use this model, please cite:

<pre><code class="language-bibtex">
@misc{{whisper_finetune_{config.model_name.lower()}}},
  title={{{{Fine-tuned Whisper Model}}}},
  author={{{{Your Name or Team Name}}}},
  year={{{{2024}}}},
  howpublished={{https://huggingface.co/{training_args.hub_model_id}}}

</code></pre>

## Acknowledgments
Thanks to the Hugging Face team and the community for providing tools to fine-tune and deploy this model.
"""

    return model_card


class ExperimentLogger:
    """Simple logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, print_metrics: bool = True, print_predictions: bool = True):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime(TIMESTAMP_FORMAT)
        
        # Create experiment directory under logs/
        self.experiment_dir = LOG_DIR / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            "metadata": {
                "name": experiment_name,
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "timeline": []
        }
        
        # Output control flags
        self.print_metrics = print_metrics
        self.print_predictions = print_predictions
    
    def _get_elapsed_time(self):
        """Get elapsed time since experiment start"""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
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
    lr_scheduler_type: str  # New
    
    # Evaluation and saving
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    logging_steps: int
    save_total_limit: int
    push_to_hub: bool
    predict_with_generate: bool  # New
    load_best_model_at_end: bool  # New
    metric_for_best_model: str  # New
    greater_is_better: bool  # New
    report_to: List[str]  # New
    
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

class EnhancedWhisperTrainer(Seq2SeqTrainer):
    """EXPERIMENTAL: Enhanced trainer with gradient monitoring and improved generation control.
    
    This is an experimental extension of Seq2SeqTrainer that adds:
    - Detailed gradient monitoring and statistics
    - Improved generation control
    - Parameter-specific training dynamics
    
    Note: Currently not in use, kept for future development.
    
    Args:
        config (ExperimentConfig): Configuration object containing monitoring settings
        logger (ExperimentLogger): Logger for tracking detailed gradient statistics
    """
    
    def __init__(self, *args, config: ExperimentConfig, logger: ExperimentLogger, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_history = defaultdict(lambda: deque(maxlen=config.gradient_history_size))
        self.config = config
        self.logger = logger
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Enhanced prediction step that properly handles both training and evaluation scenarios."""
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Get model outputs with logits
            outputs = model(**inputs)
            loss = outputs.loss if "labels" in inputs else None
            logits = outputs.logits
            
            # For evaluation with generate=True, also get the generated tokens
            if not prediction_loss_only and self.args.predict_with_generate:
                generate_kwargs = {
                    "input_features": inputs["input_features"],
                    "max_new_tokens": self.args.generation_max_length,
                    "language": self.config.generation_language,
                    "task": self.config.generation_task,
                    "use_cache": self.config.use_cache
                }
                generated_tokens = model.generate(**generate_kwargs)
                return (loss, generated_tokens, inputs.get("labels"))
            
            # For training or non-generate evaluation, return logits
            return (loss, logits, inputs.get("labels"))
    
    def training_step(self, model, inputs):
        """Enhanced training step with gradient monitoring."""
        inputs = self._prepare_inputs(inputs)
        
        # Regular training step
        loss = super().training_step(model, inputs)
        
        # Gradient monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self.log_gradient_stats(model)
        
        return loss
    
    def log_gradient_stats(self, model):
        """Log detailed gradient statistics."""
        if not hasattr(model, "named_parameters"):
            return
            
        total_norm = 0.0
        stats = {}
        
        # Collect gradient statistics
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                stats[name] = {
                    'norm': param_norm,
                    'mean': param.grad.data.mean().item(),
                    'std': param.grad.data.std().item(),
                    'max': param.grad.data.max().item(),
                    'min': param.grad.data.min().item()
                }
                
                # Track history
                self.gradient_history[name].append(param_norm)
                # Keep only recent history
                if len(self.gradient_history[name]) > 1000:
                    self.gradient_history[name].pop(0)
        
        total_norm = total_norm ** 0.5
        
        # Log to tensorboard
        self.log(
            {
                "gradient/total_norm": total_norm,
                "gradient/max_norm": max(s['norm'] for s in stats.values()),
                "gradient/min_norm": min(s['norm'] for s in stats.values()),
            }
        )
        
        # Detailed logging to console
        self.logger.log(f"\nGradient Stats (Step {self.state.global_step}):")
        self.logger.log(f"Total gradient norm: {total_norm:.4f}")
        
        # Log top N largest gradients (configurable)
        top_grads = sorted(stats.items(), key=lambda x: x[1]['norm'], reverse=True)[:self.config.log_top_n_gradients]
        self.logger.log("\nTop N largest gradients:")
        for name, stat in top_grads:
            self.logger.log(f"{name}:")
            self.logger.log(f"  Norm: {stat['norm']:.4f}")
            self.logger.log(f"  Mean: {stat['mean']:.4f}")
            self.logger.log(f"  Std:  {stat['std']:.4f}")
            
        # Parameter-specific monitoring
        self.log_parameter_dynamics(stats)
    
    def log_parameter_dynamics(self, current_stats):
        """Monitor parameter-specific training dynamics."""
        for name, history in self.gradient_history.items():
            if len(history) > 1:  # Need at least 2 points for trend
                recent_trend = history[-1] - history[-2]
                current_stat = current_stats.get(name, {})
                
                if abs(recent_trend) > self.config.significant_change_threshold:
                    self.logger.log(f"\nSignificant gradient change in {name}:")
                    self.logger.log(f"  Current norm: {current_stat.get('norm', 0):.4f}")
                    self.logger.log(f"  Change: {recent_trend:.4f}")
                    self.logger.log(f"  Statistics:")
                    self.logger.log(f"    Mean: {current_stat.get('mean', 0):.4f}")
                    self.logger.log(f"    Std:  {current_stat.get('std', 0):.4f}")


    
    def save_prediction(self, reference: str, prediction: str, print_to_terminal: bool = None):
        """Save a prediction example to the timeline"""
        elapsed = self._get_elapsed_time()
        
        # Determine whether to print based on instance setting or override
        should_print = print_to_terminal if print_to_terminal is not None else self.print_predictions
        if should_print:
            print(f"[+{elapsed}] New prediction:")
            print(f"Reference: {reference}")
            print(f"Prediction: {prediction}")
        
        self.results["timeline"].append({
            "type": "prediction",
            "elapsed": elapsed,
            "reference": reference,
            "prediction": prediction
        })
        self._save_yaml()
    
    def _save_yaml(self):
        """Save current results to YAML file"""
        yaml_path = self.experiment_dir / "results.yaml"
        with open(self.experiment_dir / "results.yaml", "w") as f:
            yaml.dump(self.results, f, default_flow_style=False)


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
            self.tb_writer = SummaryWriter(log_dir=args.logging_dir)
            self.experiment_logger.log("Initialized TensorBoard writer")
    
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
    # Load configuration
    config = ExperimentConfig.from_yaml(default_config)
    # Initialize logger
    logger = ExperimentLogger(config.name)
    
    # Log experiment initialization
    logger.log("=" * 50)
    logger.log("EXPERIMENT INITIALIZATION")
    logger.log("=" * 50)
    logger.log(f"Name: {config.name}")
    logger.log(f"Description: {config.description}")
    logger.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # Authenticate and load dataset
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("Please set the HF_TOKEN environment variable")
    login(token)
    
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
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    run_output_dir = OUTPUT_DIR / f"{config.name}_{timestamp}"
    run_output_dir.mkdir(exist_ok=True)

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
        push_to_hub=False,  # Disable automatic push
        lr_scheduler_type=config.lr_scheduler_type,
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
    
    # Add this code right before trainer.train()
    if config.push_to_hub:
        logger.log("\nConfiguring Hugging Face Hub settings...")
        
        # Verify HF token
        if not HfFolder.get_token():
            raise ValueError(
                "No Hugging Face token found. Please run `huggingface-cli login` "
                "or set the HF_TOKEN environment variable."
            )
        
        # Create the model repository
        repo = Repository(
            local_dir=run_output_dir,
            clone_from=training_args.hub_model_id,
            use_auth_token=True
        )
        
        # Initialize the repository
        repo.git_pull()

    # Fine-tune the model
    logger.log("\nStarting fine-tuning...")
    trainer.train()

    
    # Evaluate fine-tuned model
    logger.log("\nEvaluating fine-tuned model...")
    finetuned_wer = evaluate_model(model, dataset, processor)
    
    # Save results
    logger.log("\nSaving WER to txt")
    # Save results with timestamp in the filename
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    results_file = LOG_DIR / f"{config.name}_{timestamp}_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Baseline WER: {baseline_wer:.4f}\n")
        f.write(f"Fine-tuned WER: {finetuned_wer:.4f}\n")
    
    logger.log("\nExperiment complete!")
    logger.log(f"Baseline WER: {baseline_wer:.4f}")
    logger.log(f"Fine-tuned WER: {finetuned_wer:.4f}") 

    
    # After training and evaluation is complete
    if config.push_to_hub:
        # Validate hub_model_id
        if not training_args.hub_model_id:
            logger.log("Error: No hub_model_id specified in configuration")
            logger.log("Skipping upload to Hugging Face Hub")
            sys.exit(1)
        
        # Recheck token
        token = HfFolder.get_token()
        if not token:
            logger.log("Error: HF_TOKEN not found or invalid")
            logger.log("Please run `huggingface-cli login` or set the HF_TOKEN environment variable")
            sys.exit(1)

        logger.log("\nTraining completed. Would you like to upload the model to Hugging Face Hub?")
        logger.log(f"  Model Performance:")
        logger.log(f"  - Baseline WER: {baseline_wer:.4f}")
        logger.log(f"  - Fine-tuned WER: {finetuned_wer:.4f}")
        logger.log(f"  Repository: {training_args.hub_model_id}")
        
        response = input("[y/N]: ").lower().strip()
        if response == 'y':
            try:
                logger.log("\nPreparing for upload to Hugging Face Hub...")
                
                # Create repo only when ready to upload
                from huggingface_hub import HfApi
                api = HfApi()
                
                # Extract repo_id from hub_model_id
                repo_id = training_args.hub_model_id
                logger.log(f"Creating repository: {repo_id}")
                
                # Create the repo
                api.create_repo(
                    repo_id=repo_id,
                    private=True,  # Default to private for safety
                    exist_ok=True  # In case repo exists but is empty
                )
                
                # Create and save model card using existing function
                model_card = create_model_card(
                    config=config,
                    dataset=dataset,
                    baseline_wer=baseline_wer,
                    finetuned_wer=finetuned_wer,
                    training_args=training_args
                )
                
                with open(run_output_dir / "README.md", "w") as f:
                    f.write(model_card)
                
                # Now push to hub
                logger.log("Uploading model to Hugging Face Hub...")
                trainer.push_to_hub(
                    commit_message=f"Training completed - WER: {finetuned_wer:.4f}",
                    blocking=True
                )
                logger.log(f"Model successfully uploaded to: {repo_id}")
                logger.log(f"View your model at: https://huggingface.co/{repo_id}")
                
            except Exception as e:
                logger.log(f"Error during upload: {str(e)}")
                logger.log("Model saved locally but upload failed")
                sys.exit(1)
        else:
            logger.log("Upload cancelled. Model saved locally only")


