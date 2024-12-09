# Whisper Fine-tuning Experiments

This directory contains all experiments for fine-tuning Whisper models on throat microphone data.

## Structure
```
experiments/
├── experiment_001/       # Initial baseline with Whisper Base
│   ├── config.yaml      # Experiment configuration
│   └── results/         # Training logs and metrics
└── experiment_002/      # Optimized setup with Whisper Large-v3
    ├── config.yaml      # Experiment configuration
    └── results/         # Training logs and metrics
```

## Experiments Overview

### Experiment 001 (2024-12-09)
- Initial experiment with Whisper Base model
- Dataset: 180 examples
- Results: 41.67% WER improvement (101.43% → 59.76%)
- Status: Completed

### Experiment 002 (2024-12-09)
- Upgraded to Whisper Large-v3 with optimizations
- Dataset: 506 examples (363 train, 41 val, 102 test)
- Status: Ready for execution

## Documentation
Each experiment contains:
1. `config.yaml`: Full experiment configuration
2. `results/`: Training logs and metrics

See root `experiment_results.md` for comprehensive analysis.
``` 