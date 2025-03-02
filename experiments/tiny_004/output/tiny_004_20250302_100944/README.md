---
library_name: transformers
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: throatmic_subvocalization_whisper_tiny
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# throatmic_subvocalization_whisper_tiny

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3807
- Wer: 0.6449

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

| Training Loss | Epoch   | Step | Validation Loss | Wer    |
|:-------------:|:-------:|:----:|:---------------:|:------:|
| 7.2094        | 0.4464  | 25   | 5.9584          | 1.7658 |
| 4.7209        | 0.8929  | 50   | 3.3913          | 1.1397 |
| 2.5229        | 1.3393  | 75   | 2.1995          | 0.9069 |
| 1.8454        | 1.7857  | 100  | 1.8676          | 0.8454 |
| 1.6015        | 2.2321  | 125  | 1.7199          | 0.7794 |
| 1.3786        | 2.6786  | 150  | 1.6296          | 0.7574 |
| 1.2147        | 3.125   | 175  | 1.5654          | 0.7432 |
| 1.0976        | 3.5714  | 200  | 1.5200          | 0.7135 |
| 1.0156        | 4.0179  | 225  | 1.4829          | 0.6759 |
| 0.8611        | 4.4643  | 250  | 1.4689          | 0.7050 |
| 0.8818        | 4.9107  | 275  | 1.4394          | 0.6585 |
| 0.7822        | 5.3571  | 300  | 1.4273          | 0.6669 |
| 0.6969        | 5.8036  | 325  | 1.4159          | 0.6481 |
| 0.7037        | 6.25    | 350  | 1.4057          | 0.6533 |
| 0.6555        | 6.6964  | 375  | 1.3991          | 0.6475 |
| 0.5759        | 7.1429  | 400  | 1.3927          | 0.6546 |
| 0.5217        | 7.5893  | 425  | 1.3936          | 0.6397 |
| 0.5731        | 8.0357  | 450  | 1.3849          | 0.6436 |
| 0.4753        | 8.4821  | 475  | 1.3839          | 0.6345 |
| 0.4799        | 8.9286  | 500  | 1.3816          | 0.6546 |
| 0.4369        | 9.375   | 525  | 1.3824          | 0.6429 |
| 0.4424        | 9.8214  | 550  | 1.3828          | 0.6404 |
| 0.4206        | 10.2679 | 575  | 1.3888          | 0.6371 |
| 0.3735        | 10.7143 | 600  | 1.3807          | 0.6449 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
