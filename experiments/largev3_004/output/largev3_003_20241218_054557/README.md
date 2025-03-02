---
language: en
tags:
  - whisper
  - audio
  - speech-recognition
  - pytorch
  - throat-microphone
  - subvocalization
license: mit
datasets:
  - pauljunsukhan/throatmic_codered
metrics:
  - wer
model-index:
  - name: None
    results:
      - task: 
          name: Automatic Speech Recognition
          type: automatic-speech-recognition
        dataset:
          name: pauljunsukhan/throatmic_codered
          type: pauljunsukhan/throatmic_codered
        metrics:
          - name: WER
            type: wer
            value: 0.2506
---

# Whisper Fine-tuned Model

This model is a fine-tuned version of `openai/whisper-large-v3` on `pauljunsukhan/throatmic_codered`.

## Model Description
- **Model Type:** Fine-tuned Whisper model for speech recognition
- **Language:** English
- **Task:** Automatic Speech Recognition
- **Domain:** Throat Microphone Speech Recognition

## Training Details
- **Base Model:** `openai/whisper-large-v3`
- **Dataset:** `pauljunsukhan/throatmic_codered`
- **Training Examples:** 480
- **Test Examples:** 124
- **Training Steps:** 800

### Training Hyperparameters
The following hyperparameters were used during training:
- **Learning Rate:** 5e-06
- **Train Batch Size:** 16
- **Eval Batch Size:** 64
- **Seed:** 42
- **Optimizer:** AdamW with betas=(0.9,0.999) and epsilon=1e-08
- **LR Scheduler Type:** linear
- **Warmup Steps:** 50
- **Training Steps:** 800
- **Mixed Precision Training:** Native AMP
- **Weight Decay:** 0.0
- **Gradient Checkpointing:** False
- **FP16:** True
- **Label Smoothing:** 0.0
- **Max Gradient Norm:** 1.0

### Framework Versions
- **Transformers:** 4.45.2
- **PyTorch:** 2.5.1+cu124
- **Datasets:** 2.15.0
- **Tokenizers:** 0.20.3

## Training Results
| Training Loss | Epoch | Step | Validation Loss | WER |
|--------------|--------|------|-----------------|-----|
| 3.8126 | 0.8333 | 25 | 0.0000 | 0.0000 |
| 0.0000 | 0.8333 | 25 | 2.6280 | 0.5000 |
| 2.0125 | 1.6667 | 50 | 0.0000 | 0.0000 |
| 0.0000 | 1.6667 | 50 | 1.5689 | 0.3200 |
| 0.8413 | 2.5000 | 75 | 0.0000 | 0.0000 |
| 0.0000 | 2.5000 | 75 | 0.7058 | 0.2716 |
| 0.2436 | 3.3333 | 100 | 0.0000 | 0.0000 |
| 0.0000 | 3.3333 | 100 | 0.6248 | 0.2780 |
| 0.1094 | 4.1667 | 125 | 0.0000 | 0.0000 |
| 0.0000 | 4.1667 | 125 | 0.6513 | 0.2678 |
| 0.0414 | 5.0000 | 150 | 0.0000 | 0.0000 |
| 0.0000 | 5.0000 | 150 | 0.6778 | 0.2665 |
| 0.0197 | 5.8333 | 175 | 0.0000 | 0.0000 |
| 0.0000 | 5.8333 | 175 | 0.6993 | 0.2729 |
| 0.0113 | 6.6667 | 200 | 0.0000 | 0.0000 |
| 0.0000 | 6.6667 | 200 | 0.7703 | 0.2684 |
| 0.0075 | 7.5000 | 225 | 0.0000 | 0.0000 |
| 0.0000 | 7.5000 | 225 | 0.7701 | 0.2761 |
| 0.0047 | 8.3333 | 250 | 0.0000 | 0.0000 |
| 0.0000 | 8.3333 | 250 | 0.7619 | 0.2525 |
| 0.0022 | 9.1667 | 275 | 0.0000 | 0.0000 |
| 0.0000 | 9.1667 | 275 | 0.8213 | 0.2615 |
| 0.0014 | 10.0000 | 300 | 0.0000 | 0.0000 |
| 0.0000 | 10.0000 | 300 | 0.8264 | 0.2557 |
| 0.0008 | 10.8333 | 325 | 0.0000 | 0.0000 |
| 0.0000 | 10.8333 | 325 | 0.8351 | 0.2519 |
| 0.0006 | 11.6667 | 350 | 0.0000 | 0.0000 |
| 0.0000 | 11.6667 | 350 | 0.8504 | 0.2506 |
| 0.0006 | 12.5000 | 375 | 0.0000 | 0.0000 |
| 0.0000 | 12.5000 | 375 | 0.8554 | 0.2506 |
| 0.0005 | 13.3333 | 400 | 0.0000 | 0.0000 |
| 0.0000 | 13.3333 | 400 | 0.8621 | 0.2506 |
| 0.0005 | 14.1667 | 425 | 0.0000 | 0.0000 |
| 0.0000 | 14.1667 | 425 | 0.8657 | 0.2532 |
| 0.0004 | 15.0000 | 450 | 0.0000 | 0.0000 |
| 0.0000 | 15.0000 | 450 | 0.8703 | 0.2564 |
| 0.0004 | 15.8333 | 475 | 0.0000 | 0.0000 |
| 0.0000 | 15.8333 | 475 | 0.8750 | 0.2589 |
| 0.0000 | 15.8333 | 475 | 0.0000 | 0.0000 |


## Performance
- **Baseline WER:** 0.7462
- **Fine-tuned WER:** 0.2506
- **Best WER:** 0.2506 (Step 350)

## Usage

You can use this model as follows:

<pre><code class="language-python">
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load processor and model
processor = WhisperProcessor.from_pretrained("None")
model = WhisperForConditionalGeneration.from_pretrained("None")

# Example usage
inputs = processor("Audio input data", return_tensors="pt", sampling_rate=16000)
outputs = model.generate(inputs["input_features"])
transcription = processor.batch_decode(outputs, skip_special_tokens=True)
print(transcription)
</code></pre>

## Citation
If you use this model, please cite:

<pre><code class="language-bibtex">
@misc{whisper_finetune_openai_whisper_large_v3},
  title={Fine-tuned Whisper Model},
  author={Your Name or Team Name},
  year={2024},
  howpublished={https://huggingface.co/None}
</code></pre>

## Acknowledgments
Thanks to the Hugging Face team and the community for providing tools to fine-tune and deploy this model.
