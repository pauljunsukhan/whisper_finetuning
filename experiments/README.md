# Whisper Fine-tuning Experiments

OMP_NUM_THREADS=14 torchrun --nproc_per_node=2 -m experiments.experiment_002.main
 (root directory)


## Directory Structure
```
experiments/
├── README.md                    # This file
├── baseline_001/               # First baseline experiment (180 examples)
│   ├── config.yaml             # Experiment configuration
│   ├── metrics.json            # Raw metrics
│   └── results.md              # Detailed analysis
├── experiment_002/             # Future experiment
└── templates/                  # Templates for new experiments
    ├── config_template.yaml
    └── results_template.md
```

## Naming Convention
- `baseline_001`: Initial experiment with 180 examples
- `experiment_002`: Next experiment (e.g., with 1000 examples)
- Use sequential numbering for tracking

## Adding New Experiments
1. Copy templates from `templates/`
2. Create new directory with appropriate name
3. Fill in configuration and results
4. Update main README.md with latest results

## Git Workflow
```bash
# When starting new experiment
git checkout -b experiment_002

# After experiment completion
git add experiments/experiment_002/
git commit -m "experiment_002: description of changes"
git checkout main
git merge experiment_002
``` 

Token 50258: <|startoftranscript|> — Correctly starts the prediction.
Token 50259: <|en|> — Language token.
Token 50360: <|transcribe|> — Task token.
Token 50364: <|notimestamps|> — Timestamps disabled.

Key Guidelines for Whisper Fine-tuning:
Data Processing:
Audio should be 16kHz sampling rate
Input features should be mel spectrograms with 80 or 128 mel bins
Proper handling of prefix tokens (<|startoftranscript|>, language, task)
Proper padding and attention mask handling
Training Configuration:
Use small batch sizes (4-8) due to memory requirements
Learning rate around 1e-5
Use gradient checkpointing for memory efficiency
Mixed precision training recommended
Teacher forcing is standard practice
Model Settings:
Disable KV cache during training
Use greedy decoding during evaluation
Maintain proper token handling (pad, bos, eos)