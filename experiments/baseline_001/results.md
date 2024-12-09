# Baseline 001 Results

## Overview
- **Date**: 2024-01-17
- **Description**: Initial baseline experiment with 180 examples
- **Key Changes**: First experiment, establishing baseline performance

## Dataset
- Total Examples: 180
- Training Split: 144
- Test Split: 36
- Audio Format: Throat microphone recordings at 16kHz

## Configuration
```yaml
training:
  batch_size: 16
  learning_rate: 1e-5
  warmup_steps: 500
  max_steps: 1000
  gradient_checkpointing: true
  fp16: true
  regularization:
    weight_decay: 0.0
    dropout: 0.0
    label_smoothing: 0.0
```

## Results

### Metrics
- Baseline WER: 101.43%
- Final WER: 59.76%
- Absolute Improvement: 41.67%

### Training Progress
- Duration: 43 minutes
- Total Steps: 1000
- Epochs: 111.11
- Final Loss: 0.0003

#### Loss Progression
```
Step   Loss     Learning Rate
0      7.8466   1.0e-5
100    3.1987   9.2e-6
200    0.6866   9.8e-6
300    0.129    9.5e-6
...    ...      ...
1000   0.0003   4.2e-8
```

### Example Predictions

#### Baseline Model
```
Reference: "One aspect of this is the lack of "high rise" hotels on the island."
Prediction: "Well, I'm expecting this is the last of high rise hotels on the island."

Reference: "Subsequently, he accused his allies of spying on him and working to harm Samoobrona."
Prediction: "Subscribe on the QSS and on the S1 website."
```

#### Fine-tuned Model
```
Reference: "One aspect of this is the lack of "high rise" hotels on the island."
Prediction: "One I have been expecting this is the lack of high rise hotels on the island."

Reference: "When this is diagonalized, the eigenvectors are chosen as the expansion coefficients."
Prediction: "When this is diagnosed, the eye concaptors are chosen as expansion-coded patients."
```

## Analysis

### Improvements
- Significant WER reduction (41.67% absolute improvement)
- Model learned some basic audio-text alignment
- Training process was stable

### Issues
- High baseline WER (101.43%)
- Final WER still high (59.76%)
- Hallucination-like behavior
- Struggles with:
  - Complex terminology
  - Proper nouns
  - Technical terms

### Root Causes
1. **Dataset Size**
   - Only 180 examples
   - Far below industry minimum (1,000-2,000)
   - Limited exposure to variations

2. **Training Dynamics**
   - Very rapid initial loss decrease
   - Potential overfitting
   - Limited data prevents proper generalization

## Next Steps
1. **Data Collection Priority**
   - Target minimum 1,000 examples
   - Ensure diversity in:
     - Speakers
     - Sentence structures
     - Technical terms
     - Speaking speeds

2. **Model Improvements** (for next experiment)
   - Reduce learning rate to 5e-6
   - Increase batch size to 32/64
   - Add regularization suite:
     - Weight decay: 0.01
     - Dropout: 0.1
     - Label smoothing: 0.1

## Technical Notes
- Training completed without technical issues
- A10 GPU memory usage was moderate
- No OOM errors or stability issues 