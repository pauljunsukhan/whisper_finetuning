---
library_name: transformers
license: apache-2.0
base_model: openai/whisper-small
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: throatmic_subvocalization_whisper_small
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# throatmic_subvocalization_whisper_small

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8621
- Wer: 0.3855

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
| 5.1486        | 0.4464 | 25   | 3.9859          | 0.8564 |
| 3.0886        | 0.8929 | 50   | 2.1745          | 0.7057 |
| 1.7689        | 1.3393 | 75   | 1.7727          | 0.4748 |
| 1.4945        | 1.7857 | 100  | 1.5883          | 0.4243 |
| 1.3174        | 2.2321 | 125  | 1.4747          | 0.3946 |
| 1.082         | 2.6786 | 150  | 1.4101          | 0.3816 |
| 0.967         | 3.125  | 175  | 1.3374          | 0.3797 |
| 0.7866        | 3.5714 | 200  | 1.2746          | 0.3700 |
| 0.6628        | 4.0179 | 225  | 1.0417          | 0.3784 |
| 0.2359        | 4.4643 | 250  | 0.7904          | 0.3577 |
| 0.1835        | 4.9107 | 275  | 0.7902          | 0.3571 |
| 0.1051        | 5.3571 | 300  | 0.8173          | 0.3726 |
| 0.1088        | 5.8036 | 325  | 0.8255          | 0.3674 |
| 0.0894        | 6.25   | 350  | 0.8310          | 0.3603 |
| 0.0683        | 6.6964 | 375  | 0.8385          | 0.3726 |
| 0.051         | 7.1429 | 400  | 0.8621          | 0.3855 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
