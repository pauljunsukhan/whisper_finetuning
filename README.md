# Throat Microphone Whisper Fine-tuning

This repository contains experiments for fine-tuning OpenAI's Whisper model on throat microphone recordings.

## Project Status
Currently in initial experimentation phase. First results show promise but indicate need for larger dataset.
See [experiment_results.md](experiment_results.md) for detailed analysis.

## Repository Structure
```
.
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── explore_dataset.py           # Dataset exploration utilities
├── experiment_results.md        # Detailed experiment analysis
├── experiments/                 # All experiment code
│   ├── README.md               # Experiments documentation
│   ├── baseline_001/           # Initial baseline experiment
│   │   ├── config.yaml         # Experiment configuration
│   │   ├── results.md          # Results and analysis
│   │   └── whisper_experiment.py
│   ├── experiment_002/         # Current experiment
│   │   ├── components/         # Modular components
│   │   │   ├── base.py        # Base classes
│   │   │   ├── data.py        # Data handling
│   │   │   ├── logger.py      # Logging utilities
│   │   │   ├── model.py       # Model management
│   │   │   ├── state.py       # State management
│   │   │   └── trainer.py     # Training logic
│   │   ├── config.yaml        # Experiment configuration
│   │   └── main.py           # Entry point
│   └── templates/              # Templates for new experiments
│       ├── config_template.yaml
│       └── results_template.md
└── outputs/                    # Experiment outputs
    └── experiment_002/         # Current experiment outputs
        ├── config/            # Saved configurations
        └── logs/              # Experiment logs
```

## Dataset
Using [pauljunsukhan/throatmic_codered](https://huggingface.co/datasets/pauljunsukhan/throatmic_codered) from Hugging Face.

## Current Results
- Baseline WER: 101.43%
- Fine-tuned WER: 59.76%
- Significant improvement but still room for enhancement

## Next Steps
1. Expand dataset (target: 1,000+ examples)
2. Implement model improvements
3. Run comparative experiments

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Set your Hugging Face token
export HF_TOKEN=your_token_here

# Run the experiment
python -m experiments.experiment_002.main
```

## Hardware Requirements
- Tested on Lambda Labs A10 GPU (24GB VRAM)
- CUDA compatible GPU recommended
