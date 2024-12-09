# Whisper Fine-tuning for Throat Microphone Data

This repository contains code for fine-tuning OpenAI's Whisper Large V3 model on throat microphone data.

## Project Overview

Fine-tuning Whisper for improved transcription of throat microphone recordings, with a focus on:
- Adapting to throat microphone acoustics
- Handling different audio characteristics
- Optimizing for resource constraints

## Current Status

The project is in development with some key challenges identified:
1. Memory optimization for large model training
2. Baseline performance improvements needed
3. Audio preprocessing verification

See bug reports for detailed analysis of current issues.

## Setup

### Requirements
- Python 3.9+
- CUDA-capable GPU (tested on A10 24GB)
- PyTorch 2.0+
- Transformers library

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd whisper_finetuning
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the main training script:
```bash
python whisper_experiment.py
```

### Configuration

Key parameters in `whisper_experiment.py`:
```python
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "pauljunsukhan/throatmic_codered"
```

Training arguments can be modified in the script:
```python
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    ...
)
```

## Dataset

Using the throat microphone dataset:
- 404 training samples
- 102 test samples
- Audio sampled at 16kHz
- Various speakers and conditions

## Known Issues

1. Memory Management:
   - Currently hits OOM on 24GB GPU
   - Needs larger GPU or memory optimizations

2. Performance:
   - High baseline WER (0.7899)
   - Audio preprocessing needs verification

See bug reports for detailed analysis.

## Directory Structure

```
.
├── whisper_experiment.py    # Main training script
├── requirements.txt         # Python dependencies
├── bug_reports/            # Analysis and debugging
├── experiments/            # Experiment configurations
└── README.md              # This file
```

## Next Steps

1. Move to larger GPU
2. Verify audio preprocessing
3. Implement memory optimizations
4. Add data augmentation

## Contributing

Please read the bug reports before contributing. Key areas for improvement:
- Memory optimization
- Audio preprocessing
- Training configuration

## License

[Add License Information]
