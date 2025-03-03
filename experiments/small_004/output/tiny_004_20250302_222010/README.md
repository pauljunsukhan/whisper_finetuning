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
- Loss: 0.8254
- Wer: 0.3648

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
| 3.0886        | 0.8929 | 50   | 2.1748          | 0.7076 |
| 1.7689        | 1.3393 | 75   | 1.7725          | 0.4748 |
| 1.4947        | 1.7857 | 100  | 1.5882          | 0.4224 |
| 1.3175        | 2.2321 | 125  | 1.4805          | 0.3933 |
| 1.0851        | 2.6786 | 150  | 1.4118          | 0.3790 |
| 0.9714        | 3.125  | 175  | 1.3399          | 0.3816 |
| 0.7866        | 3.5714 | 200  | 1.2543          | 0.3693 |
| 0.666         | 4.0179 | 225  | 1.0375          | 0.3758 |
| 0.2356        | 4.4643 | 250  | 0.7912          | 0.3551 |
| 0.174         | 4.9107 | 275  | 0.7736          | 0.3674 |
| 0.0996        | 5.3571 | 300  | 0.8049          | 0.3803 |
| 0.0712        | 5.8036 | 325  | 0.7859          | 0.3693 |
| 0.0625        | 6.25   | 350  | 0.8147          | 0.3596 |
| 0.0464        | 6.6964 | 375  | 0.8254          | 0.3648 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
