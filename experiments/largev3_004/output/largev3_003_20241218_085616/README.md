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
            value: 0.2583
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
| 0.0000 | 0.8333 | 25 | 2.6283 | 0.5000 |
| 2.0124 | 1.6667 | 50 | 0.0000 | 0.0000 |
| 0.0000 | 1.6667 | 50 | 1.5646 | 0.3200 |
| 0.8458 | 2.5000 | 75 | 0.0000 | 0.0000 |
| 0.0000 | 2.5000 | 75 | 0.7219 | 0.2748 |
| 0.2478 | 3.3333 | 100 | 0.0000 | 0.0000 |
| 0.0000 | 3.3333 | 100 | 0.6246 | 0.2748 |
| 0.1114 | 4.1667 | 125 | 0.0000 | 0.0000 |
| 0.0000 | 4.1667 | 125 | 0.6537 | 0.2723 |
| 0.0414 | 5.0000 | 150 | 0.0000 | 0.0000 |
| 0.0000 | 5.0000 | 150 | 0.6800 | 0.2710 |
| 0.0205 | 5.8333 | 175 | 0.0000 | 0.0000 |
| 0.0000 | 5.8333 | 175 | 0.7095 | 0.2748 |
| 0.0103 | 6.6667 | 200 | 0.0000 | 0.0000 |
| 0.0000 | 6.6667 | 200 | 0.7791 | 0.2716 |
| 0.0065 | 7.5000 | 225 | 0.0000 | 0.0000 |
| 0.0000 | 7.5000 | 225 | 0.7807 | 0.2640 |
| 0.0046 | 8.3333 | 250 | 0.0000 | 0.0000 |
| 0.0000 | 8.3333 | 250 | 0.7535 | 0.2659 |
| 0.0037 | 9.1667 | 275 | 0.0000 | 0.0000 |
| 0.0000 | 9.1667 | 275 | 0.7932 | 0.2729 |
| 0.0015 | 10.0000 | 300 | 0.0000 | 0.0000 |
| 0.0000 | 10.0000 | 300 | 0.7857 | 0.2615 |
| 0.0009 | 10.8333 | 325 | 0.0000 | 0.0000 |
| 0.0000 | 10.8333 | 325 | 0.7988 | 0.2615 |
| 0.0007 | 11.6667 | 350 | 0.0000 | 0.0000 |
| 0.0000 | 11.6667 | 350 | 0.8082 | 0.2583 |
| 0.0007 | 12.5000 | 375 | 0.0000 | 0.0000 |
| 0.0000 | 12.5000 | 375 | 0.8174 | 0.2608 |
| 0.0006 | 13.3333 | 400 | 0.0000 | 0.0000 |
| 0.0000 | 13.3333 | 400 | 0.8261 | 0.2608 |
| 0.0005 | 14.1667 | 425 | 0.0000 | 0.0000 |
| 0.0000 | 14.1667 | 425 | 0.8305 | 0.2602 |
| 0.0005 | 15.0000 | 450 | 0.0000 | 0.0000 |
| 0.0000 | 15.0000 | 450 | 0.8354 | 0.2608 |
| 0.0004 | 15.8333 | 475 | 0.0000 | 0.0000 |
| 0.0000 | 15.8333 | 475 | 0.8404 | 0.2583 |
| 0.0000 | 15.8333 | 475 | 0.0000 | 0.0000 |


## Performance
- **Baseline WER:** 0.7462
- **Fine-tuned WER:** 0.2583
- **Best WER:** 0.2583 (Step 350)

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
