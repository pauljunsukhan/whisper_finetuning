---
library_name: transformers
license: mit
base_model: distil-whisper/distil-large-v3
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: throatmic_subvocalization_whisper_distil-large-v3
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# throatmic_subvocalization_whisper_distil-large-v3

This model is a fine-tuned version of [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8634
- Wer: 0.3202

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-06
- train_batch_size: 16
- eval_batch_size: 64
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 50
- training_steps: 800
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Wer    |
|:-------------:|:------:|:----:|:---------------:|:------:|
| 4.8931        | 0.4464 | 25   | 3.3920          | 0.7749 |
| 2.3126        | 0.8929 | 50   | 1.2760          | 0.4554 |
| 0.8156        | 1.3393 | 75   | 0.9349          | 0.3648 |
| 0.7011        | 1.7857 | 100  | 0.7822          | 0.3040 |
| 0.5483        | 2.2321 | 125  | 0.7531          | 0.2969 |
| 0.3743        | 2.6786 | 150  | 0.7529          | 0.3525 |
| 0.2984        | 3.125  | 175  | 0.7249          | 0.3402 |
| 0.1953        | 3.5714 | 200  | 0.7531          | 0.3182 |
| 0.1811        | 4.0179 | 225  | 0.7380          | 0.2788 |
| 0.0976        | 4.4643 | 250  | 0.7636          | 0.3234 |
| 0.1152        | 4.9107 | 275  | 0.7868          | 0.2962 |
| 0.0694        | 5.3571 | 300  | 0.8141          | 0.3072 |
| 0.0489        | 5.8036 | 325  | 0.8237          | 0.3279 |
| 0.0459        | 6.25   | 350  | 0.8634          | 0.3202 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
