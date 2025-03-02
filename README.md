# Throat Microphone Whisper Fine-tuning

This repository contains experiments for fine-tuning OpenAI's Whisper model on throat microphone recordings.

## Project Status
V3 folders in experiments have pretty verbose and featured training scripts with all the knobs you might want to twiddle. base_003 is using the whisper base model while largev3_003 uses the large_v3 model
baseline_001 was the very simple first script.
Compiling pytorch breaks whisper tuning, I think it just isn't made for the encoder/decoder architecture
Takes about 1hr and about 70GB to fine tune on H100 
The dataset used was pretty flawed, (the throat mic dataset) before the datacleaning and expansion of the latest throatmic dataset generator.

There is an extremely annoying warning that keeps popping up during traning for later versions of the transformers library, however this also means that there are some methods I'm using here which are now depricated

The attention mask missing warning during training is a lie, it is there (whisper has it built in) The terminal print logs show the attention mask to confirm.

Look at the yaml in the logs to see how training goes and you can also view the tensorflow dashboard.
I used uv 

Explore dataset is used to show the structure of the hugging face repo for my throat mic dataset. 

You'll need to save a HF_TOKEN environmental variable.

The simple baseline script is functionally almost identical to the fleshed out V3 other than the lack of logging and other knobs and dials. Two things:

The v3 does normalize the audio samples which baseline script doesn't. this can be good or bad because in normal case this will boost volume but in cases where there is clipping it could reduce it. Normalization is just an across the board volume gain, as opposed to dynamic compression. I'm not sure how much audio processing should be done, I'm sure there are papers on this...

I correctly set the transcribe, english and start of transcription/end of text tokens. For whisper start of transcription is used for both start and end tokens.

# Whisper Fine-tuning Environment

This repository contains the necessary files to set up a Python environment for fine-tuning OpenAI's Whisper models.

## Requirements

- Python 3.9.21 (recommended)
- uv package manager (will be installed by the setup scripts if not present)

## Dependencies

The main dependencies required for Whisper fine-tuning are:

- torch
- transformers
- datasets
- tokenizers
- evaluate
- huggingface-hub
- tensorboard
- tqdm
- pyyaml
- numpy

## Setup

### Linux/macOS

1. Make the setup script executable:
   ```bash
   chmod +x setup_whisper_finetune.sh
   ```

2. Run the setup script:
   ```bash
   ./setup_whisper_finetune.sh
   ```

3. Activate the virtual environment:
   ```bash
   source whisper_finetune/bin/activate
   ```


## Using the Environment

Once the environment is set up and activated, you can run the Whisper fine-tuning scripts:

```bash
python experiments/largev3_004/whisper_experiment_004.py
``` 