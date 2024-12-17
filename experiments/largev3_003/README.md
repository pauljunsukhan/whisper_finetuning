# Whisper Fine-tuning Experiment

## Prerequisites

### System Requirements
- Python 3.9+
- CUDA-capable GPU
- Git LFS (required for model upload to HuggingFace Hub)

### Installing Git LFS
On Ubuntu:
```bash
sudo apt-get update
sudo apt-get install git-lfs
```

On macOS:
```bash
brew install git-lfs
```

On Windows:
```bash
# Using chocolatey
choco install git-lfs

# Or download and install from:
# https://git-lfs.github.com
```

After installation, initialize Git LFS:
```bash
git lfs install
```

## Python Dependencies
Requirements are listed in requirements.txt

## Configuration
See config.yaml for experiment configuration options.

## Running the Experiment
```bash
python whisper_experiment_003.py
``` 