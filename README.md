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

I don't actually think the simple baseline script is functionally any different than the fleshed out V3 other than the lack of logging and other knobs and dials. The v3 does normalize the audio samples which baseline script doesn't. this can be good or bad because in normal case this will boost volume but in cases where there is clipping it could reduce it. Normalization is just an across the board volume gain, as opposed to dynamic compression. I'm not sure how much audio processing should be done, I'm sure there are papers on this...