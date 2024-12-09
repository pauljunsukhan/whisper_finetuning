# Experiment 002

## Overview
- Date: 2024-12-09
- Description: Fine-tuning Whisper Large-v3 on throat microphone data with simplified approach
- Key Changes: 
  - Upgraded to Large-v3 model
  - Simplified preprocessing pipeline
  - Focused on core functionality
  - Standard HuggingFace training setup
  - Direct use of Whisper's processor

## Setup
- Model: openai/whisper-large-v3
- Dataset Size: 506 examples
- Hardware: Lambda Labs A10 (24GB)
- Framework: HuggingFace Transformers

## Key Configurations

1. Training Parameters:
   - Per Device Batch Size: 8
   - Gradient Accumulation Steps: 4
   - Effective Batch Size: 32
   - Learning Rate: 1e-5
   - Number of Epochs: 10
   - Early Stopping:
     - Patience: 3 epochs
     - Threshold: 0.01 WER

2. Model Configuration:
   - Mixed Precision (FP16)
   - Gradient Checkpointing: Enabled
   - Generation Max Length: 225
   - Label Smoothing: 0.1
   - Weight Decay: 0.01
   - Warmup Ratio: 0.1

3. Data Processing:
   - Direct audio input to Whisper processor
   - Standard padding and collation
   - No custom preprocessing
   - Sampling Rate: 16kHz

4. Evaluation:
   - Primary Metric: WER
   - Evaluation Strategy: Steps
   - Eval Steps: 100
   - Save Steps: 100
   - Logging Steps: 50
   - Save Total Limit: 3

## Monitoring
1. Training Metrics:
   - Word Error Rate (WER)
   - Training Loss
   - Learning Rate
   - Step Times

## Results
- Baseline WER: TBD
- Final WER: TBD
- Training Time: TBD
- Best Checkpoint: TBD

## Analysis

1. Strengths:
   - Simple, maintainable codebase
   - Standard HuggingFace pipeline
   - Reliable training setup
   - Easy to reproduce
   - Clear configuration

2. Limitations:
   - No custom audio preprocessing
   - Standard batch sizes
   - Basic monitoring

3. Observations:
   - TBD (after running)

## Next Steps
1. Run baseline evaluation
2. Monitor training progress
3. Evaluate WER improvements
4. Consider custom preprocessing if needed
5. Document training outcomes