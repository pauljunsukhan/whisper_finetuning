# Experiment Templates

This directory contains templates for creating new Whisper fine-tuning experiments.

## Templates

### `config.yaml.template`
Standard configuration template for experiments:
- Experiment metadata
- Environment settings
- Dataset configuration
- Training parameters
- Generation settings
- Evaluation metrics

### `README.md.template`
Documentation template for experiment analysis:
- Overview and goals
- Setup details
- Key configurations
- Results and metrics
- Analysis and observations
- Next steps

## Usage

1. Create new experiment directory:
```bash
mkdir experiments/experiment_XXX
```

2. Copy templates:
```bash
cp templates/config.yaml.template experiments/experiment_XXX/config.yaml
```

3. Update configuration:
- Set experiment name and date
- Configure model and dataset
- Adjust training parameters
- Set evaluation metrics

4. Run experiment:
```bash
python whisper_experiment.py
```

5. Document results in `experiment_results.md`

## Template Structure

### Configuration Template
```yaml
experiment:
  name: "experiment_XXX"
  date: "YYYY-MM-DD"
  description: "..."

environment:
  platform: "Lambda Labs A10"
  gpu_memory: "24GB"
  ...

dataset:
  source: "pauljunsukhan/throatmic_codered"
  ...

training:
  batch_size: ...
  learning_rate: ...
  ...

evaluation:
  primary_metric: "WER"
  ...
```

### Documentation Template
```markdown
# Experiment XXX

## Overview
- Date: YYYY-MM-DD
- Description: ...
- Key Changes: ...

## Results
- Baseline WER: XXX%
- Final WER: XXX%
- Improvement: XXX%

...