from datasets import load_dataset
import numpy as np
from huggingface_hub import login
import os
from collections import defaultdict
import requests

# Increase timeout for requests
requests.adapters.DEFAULT_RETRIES = 5
requests.adapters.DEFAULT_TIMEOUT = 30

# Authenticate with Hugging Face
token = os.getenv("HF_TOKEN")
if token is None:
    raise ValueError("Please set the HF_TOKEN environment variable")
login(token)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("pauljunsukhan/throatmic_codered", token=token)
print("\nDataset structure:")
print(f"Keys in dataset: {dataset.keys()}")

# Dataset statistics
print("\nDataset Statistics:")
print(f"Total examples: {len(dataset['train'])}")

# Examine audio characteristics
durations = []
sampling_rates = set()
print("\nAudio Characteristics:")
for example in dataset['train']:
    audio = example['audio']
    durations.append(len(audio['array']) / audio['sampling_rate'])
    sampling_rates.add(audio['sampling_rate'])

print(f"Sampling rates: {sampling_rates}")
print(f"Average duration: {np.mean(durations):.2f}s")
print(f"Min duration: {np.min(durations):.2f}s")
print(f"Max duration: {np.max(durations):.2f}s")

# Text characteristics
word_counts = []
char_counts = []
for example in dataset['train']:
    text = example['text']
    words = text.split()
    word_counts.append(len(words))
    char_counts.append(len(text))

print("\nText Characteristics:")
print(f"Average words per example: {np.mean(word_counts):.1f}")
print(f"Average characters per example: {np.mean(char_counts):.1f}")

# Print a few examples
print("\nFirst 5 examples:")
for i in range(min(5, len(dataset['train']))):
    print(f"\nExample {i+1}:")
    print(f"Text: {dataset['train'][i]['text']}")
    audio = dataset['train'][i]['audio']
    print(f"Duration: {len(audio['array']) / audio['sampling_rate']:.2f}s") 