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
- Loss: 0.8759
- Wer: 0.2595

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
| 3.8127        | 0.8333  | 25   | 2.6281          | 0.5006 |
| 2.0125        | 1.6667  | 50   | 1.5651          | 0.3206 |
| 0.8455        | 2.5     | 75   | 0.7154          | 0.2723 |
| 0.2471        | 3.3333  | 100  | 0.6245          | 0.2742 |
| 0.1106        | 4.1667  | 125  | 0.6545          | 0.2684 |
| 0.0413        | 5.0     | 150  | 0.6800          | 0.2691 |
| 0.0204        | 5.8333  | 175  | 0.7022          | 0.2761 |
| 0.0103        | 6.6667  | 200  | 0.7689          | 0.2615 |
| 0.0063        | 7.5     | 225  | 0.7760          | 0.2595 |
| 0.005         | 8.3333  | 250  | 0.7768          | 0.2634 |
| 0.0043        | 9.1667  | 275  | 0.8041          | 0.2615 |
| 0.0036        | 10.0    | 300  | 0.7596          | 0.2583 |
| 0.0012        | 10.8333 | 325  | 0.8214          | 0.2646 |
| 0.0008        | 11.6667 | 350  | 0.8420          | 0.2589 |
| 0.0007        | 12.5    | 375  | 0.8447          | 0.2583 |
| 0.0006        | 13.3333 | 400  | 0.8523          | 0.2583 |
| 0.0005        | 14.1667 | 425  | 0.8566          | 0.2564 |
| 0.0005        | 15.0    | 450  | 0.8612          | 0.2570 |
| 0.0005        | 15.8333 | 475  | 0.8661          | 0.2615 |
| 0.0004        | 16.6667 | 500  | 0.8693          | 0.2608 |
| 0.0004        | 17.5    | 525  | 0.8729          | 0.2595 |
| 0.0004        | 18.3333 | 550  | 0.8759          | 0.2595 |


### Framework versions

- Transformers 4.45.2
- Pytorch 2.5.1+cu124
- Datasets 2.15.0
- Tokenizers 0.20.3
