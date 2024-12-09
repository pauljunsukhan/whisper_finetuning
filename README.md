# Throat Microphone Whisper Fine-tuning

This repository contains experiments for fine-tuning OpenAI's Whisper model on throat microphone recordings.

## Project Status
Currently in initial experimentation phase. First results show promise but indicate need for larger dataset.
See [experiment_results.md](experiment_results.md) for detailed analysis.

## Repository Structure
```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── whisper_experiment.py     # Main training script
├── experiment_results.md     # Detailed experiment analysis
├── experiment_results.txt    # Raw metrics
└── whisper_finetuned/       # Model checkpoints and logs
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
python whisper_experiment.py
```

## Hardware Requirements
- Tested on Lambda Labs A10 GPU (24GB VRAM)
- CUDA compatible GPU recommended
