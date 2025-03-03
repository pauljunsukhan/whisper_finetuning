---
library_name: transformers
license: mit
base_model: distil-whisper/distil-small.en
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: throatmic_subvocalization_whisper_distil-small.en
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# throatmic_subvocalization_whisper_distil-small.en

This model is a fine-tuned version of [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2105
- Wer: 0.3842

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
| 4.4036        | 0.4464 | 25   | 3.9640          | 0.9133 |
| 3.4086        | 0.8929 | 50   | 2.6473          | 0.6824 |
| 1.9018        | 1.3393 | 75   | 1.7261          | 0.5239 |
| 1.2853        | 1.7857 | 100  | 1.3489          | 0.4916 |
| 1.0448        | 2.2321 | 125  | 1.2424          | 0.4508 |
| 0.8227        | 2.6786 | 150  | 1.1834          | 0.4295 |
| 0.6852        | 3.125  | 175  | 1.1434          | 0.4172 |
| 0.5233        | 3.5714 | 200  | 1.1203          | 0.4082 |
| 0.4956        | 4.0179 | 225  | 1.0898          | 0.3965 |
| 0.3188        | 4.4643 | 250  | 1.0860          | 0.3907 |
| 0.347         | 4.9107 | 275  | 1.0959          | 0.3887 |
| 0.2575        | 5.3571 | 300  | 1.1005          | 0.3849 |
| 0.195         | 5.8036 | 325  | 1.1066          | 0.3752 |
| 0.1721        | 6.25   | 350  | 1.1233          | 0.3829 |
| 0.1611        | 6.6964 | 375  | 1.1312          | 0.3739 |
| 0.1173        | 7.1429 | 400  | 1.1490          | 0.3803 |
| 0.0955        | 7.5893 | 425  | 1.1740          | 0.3855 |
| 0.1005        | 8.0357 | 450  | 1.1826          | 0.3836 |
| 0.0578        | 8.4821 | 475  | 1.2173          | 0.3939 |
| 0.069         | 8.9286 | 500  | 1.2105          | 0.3842 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
