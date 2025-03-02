---
library_name: transformers
license: apache-2.0
base_model: openai/whisper-base
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: throatmic_subvocalization_whisper_base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# throatmic_subvocalization_whisper_base

This model is a fine-tuned version of [openai/whisper-base](https://huggingface.co/openai/whisper-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1228
- Wer: 0.5459

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
| 7.8587        | 0.4464 | 25   | 6.9278          | 1.3758 |
| 5.3129        | 0.8929 | 50   | 3.6133          | 1.2219 |
| 2.3203        | 1.3393 | 75   | 1.8454          | 1.2380 |
| 1.4604        | 1.7857 | 100  | 1.4801          | 0.7549 |
| 1.2345        | 2.2321 | 125  | 1.3462          | 0.6779 |
| 1.0225        | 2.6786 | 150  | 1.2659          | 0.6481 |
| 0.8716        | 3.125  | 175  | 1.2188          | 0.6061 |
| 0.7413        | 3.5714 | 200  | 1.1807          | 0.6061 |
| 0.6884        | 4.0179 | 225  | 1.1523          | 0.5789 |
| 0.5424        | 4.4643 | 250  | 1.1423          | 0.5815 |
| 0.5548        | 4.9107 | 275  | 1.1269          | 0.5951 |
| 0.4597        | 5.3571 | 300  | 1.1245          | 0.5809 |
| 0.3878        | 5.8036 | 325  | 1.1131          | 0.5776 |
| 0.3648        | 6.25   | 350  | 1.1130          | 0.5459 |
| 0.3554        | 6.6964 | 375  | 1.1139          | 0.5621 |
| 0.294         | 7.1429 | 400  | 1.1131          | 0.5563 |
| 0.2431        | 7.5893 | 425  | 1.1235          | 0.5569 |
| 0.2668        | 8.0357 | 450  | 1.1108          | 0.5653 |
| 0.1998        | 8.4821 | 475  | 1.1228          | 0.5459 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
