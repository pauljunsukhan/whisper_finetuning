---
library_name: transformers
license: apache-2.0
base_model: openai/whisper-large-v3
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: throatmic_subvocalization_whisper
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# throatmic_subvocalization_whisper

This model is a fine-tuned version of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7952
- Wer: 0.2672

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
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 50
- training_steps: 800
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss | Wer    |
|:-------------:|:-------:|:----:|:---------------:|:------:|
| 3.8126        | 0.8333  | 25   | 2.6284          | 0.5    |
| 2.0124        | 1.6667  | 50   | 1.5652          | 0.3206 |
| 0.8472        | 2.5     | 75   | 0.7386          | 0.2735 |
| 0.2492        | 3.3333  | 100  | 0.6263          | 0.2684 |
| 0.113         | 4.1667  | 125  | 0.6507          | 0.2729 |
| 0.0412        | 5.0     | 150  | 0.6802          | 0.2678 |
| 0.019         | 5.8333  | 175  | 0.6998          | 0.2793 |
| 0.0106        | 6.6667  | 200  | 0.7997          | 0.2640 |
| 0.0066        | 7.5     | 225  | 0.7805          | 0.2754 |
| 0.0067        | 8.3333  | 250  | 0.7283          | 0.2653 |
| 0.0054        | 9.1667  | 275  | 0.7711          | 0.2665 |
| 0.002         | 10.0    | 300  | 0.7942          | 0.2761 |
| 0.0011        | 10.8333 | 325  | 0.7952          | 0.2672 |


### Framework versions

- Transformers 4.45.2
- Pytorch 2.5.1+cu124
- Datasets 2.15.0
- Tokenizers 0.20.3
