from datasets import load_dataset
import numpy as np
from huggingface_hub import login
import os

# Authenticate with Hugging Face
token = os.getenv("HF_TOKEN")
if token is None:
    raise ValueError("Please set the HF_TOKEN environment variable")
login(token)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("pauljunsukhan/throatmic_codered")
print("\nDataset structure:")
print(f"Keys in dataset: {dataset.keys()}")

# Examine first example
print("\nExamining first example:")
example = dataset['train'][0]
print("\nKeys in example:")
for key in example:
    if key == 'audio':
        audio_data = example[key]
        print(f"\naudio:")
        print(f"  - sampling_rate: {audio_data['sampling_rate']} Hz")
        print(f"  - array shape: {audio_data['array'].shape}")
        print(f"  - array dtype: {audio_data['array'].dtype}")
        print(f"  - duration: {len(audio_data['array']) / audio_data['sampling_rate']:.2f} seconds")
    else:
        print(f"\n{key}: {example[key]}")

# Print some statistics
print("\nDataset statistics:")
print(f"Number of examples: {len(dataset['train'])}")

# Print a few text examples
print("\nFirst 5 transcriptions:")
for i in range(min(5, len(dataset['train']))):
    print(f"{i+1}. {dataset['train'][i]['text']}") 