# Whisper Fine-tuning Experiments

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