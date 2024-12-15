# Experiment Results: {config.name}

## Experiment Details
- **Date**: {config.date}
- **Description**: {config.description}
- **Model**: {MODEL_NAME}

## Dataset Information
- **Source**: {DATASET_NAME}
- **Train Split**: {len(dataset['train'])} examples
- **Test Split**: {len(dataset['test'])} examples

## Training Configuration
- **Batch Size**: {config.batch_size}
- **Learning Rate**: {config.learning_rate}
- **Max Steps**: {config.max_steps}
- **Warmup Steps**: {config.warmup_steps}

### Regularization
- **Weight Decay**: {config.weight_decay}
- **Dropout**: {config.dropout}
- **Label Smoothing**: {config.label_smoothing}

## Results
- **Baseline WER**: {baseline_wer:.4f}
- **Final WER**: {finetuned_wer:.4f}
- **Improvement**: {baseline_wer - finetuned_wer:.4f}

## Training Curves
[Include tensorboard screenshots here]

## Example Predictions
[Will be populated during evaluation]
