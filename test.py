from transformers import WhisperProcessor

# Load the processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Define language and task tokens
lang_token = "<|en|>"  # English
task_token = "<|transcribe|>"  # Transcription task

# Convert tokens to IDs
lang_id = processor.tokenizer.convert_tokens_to_ids(lang_token)
task_id = processor.tokenizer.convert_tokens_to_ids(task_token)

print(f"Language token ID for 'en': {lang_id}")
print(f"Task token ID for 'transcribe': {task_id}")

bos_token_id = processor.tokenizer.bos_token_id
bos_token = processor.tokenizer.convert_ids_to_tokens(bos_token_id)
print(f"BOS Token ID: {bos_token_id}")
print(f"BOS Token: {bos_token}")


# Define forced decoder IDs
if lang_id is not None and task_id is not None:
    forced_decoder_ids = [
        (1, lang_id),
        (2, task_id),
    ]
    print(f"Forced decoder IDs: {forced_decoder_ids}")
else:
    print("Error: Failed to retrieve token IDs for language or task.")
