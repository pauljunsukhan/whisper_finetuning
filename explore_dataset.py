from datasets import load_dataset
import numpy as np
from huggingface_hub import login
import os
import yaml
from pathlib import Path
import librosa
from tqdm import tqdm

def analyze_audio_stats(dataset):
    """Analyze audio statistics across the dataset"""
    durations = []
    sampling_rates = []
    array_shapes = []
    
    print("\nAnalyzing audio statistics...")
    for example in tqdm(dataset, desc="Processing examples"):
        audio = example['audio']
        duration = len(audio['array']) / audio['sampling_rate']
        durations.append(duration)
        sampling_rates.append(audio['sampling_rate'])
        array_shapes.append(len(audio['array']))
    
    return {
        'duration': {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations)
        },
        'sampling_rate': {
            'unique': list(set(sampling_rates))
        },
        'array_shape': {
            'mean': np.mean(array_shapes),
            'std': np.std(array_shapes),
            'min': np.min(array_shapes),
            'max': np.max(array_shapes)
        }
    }

def analyze_text_stats(dataset):
    """Analyze text statistics across the dataset"""
    word_counts = []
    char_counts = []
    
    print("\nAnalyzing text statistics...")
    for example in tqdm(dataset, desc="Processing examples"):
        text = example['text']
        words = text.split()
        word_counts.append(len(words))
        char_counts.append(len(text))
    
    return {
        'word_count': {
            'mean': np.mean(word_counts),
            'std': np.std(word_counts),
            'min': np.min(word_counts),
            'max': np.max(word_counts)
        },
        'char_count': {
            'mean': np.mean(char_counts),
            'std': np.std(char_counts),
            'min': np.min(char_counts),
            'max': np.max(char_counts)
        }
    }

def main():
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