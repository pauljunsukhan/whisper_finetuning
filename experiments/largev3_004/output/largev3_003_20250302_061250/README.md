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
- Loss: 0.5656
- Wer: 0.2044

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
| 3.7651        | 0.4464 | 25   | 2.5976          | 0.5201 |
| 2.0527        | 0.8929 | 50   | 1.5234          | 0.3053 |
| 0.7656        | 1.3393 | 75   | 0.5719          | 0.2620 |
| 0.3496        | 1.7857 | 100  | 0.4706          | 0.2322 |
| 0.2532        | 2.2321 | 125  | 0.4426          | 0.2199 |
| 0.1385        | 2.6786 | 150  | 0.4657          | 0.2290 |
| 0.1041        | 3.125  | 175  | 0.4639          | 0.2096 |
| 0.0541        | 3.5714 | 200  | 0.4846          | 0.2083 |
| 0.0453        | 4.0179 | 225  | 0.4711          | 0.1973 |
| 0.0182        | 4.4643 | 250  | 0.5187          | 0.2154 |
| 0.0257        | 4.9107 | 275  | 0.5158          | 0.2128 |
| 0.0113        | 5.3571 | 300  | 0.5966          | 0.2141 |
| 0.0067        | 5.8036 | 325  | 0.5647          | 0.2109 |
| 0.0086        | 6.25   | 350  | 0.5656          | 0.2044 |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
