import os
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
from huggingface_hub import login
from tqdm import tqdm

# Constants
MODEL_NAME = "openai/whisper-base"
DATASET_NAME = "pauljunsukhan/throatmic_codered"
OUTPUT_DIR = "whisper_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

def prepare_dataset(batch, idx):
    try:
        print(f"Processing example {idx}...")
        # Load and resample audio
        audio = batch["audio"]
        
        # Process audio - ensure we're using 16kHz which is what Whisper expects
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        # Process text
        batch["labels"] = processor.tokenizer(batch["text"], truncation=True).input_ids
        batch["input_features"] = input_features
        
        return batch
    except Exception as e:
        print(f"Error processing example {idx}: {str(e)}")
        raise e

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Get input_features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.input_ids == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def evaluate_model(model, dataset, processor, split="test"):
    print(f"\nEvaluating model on {split} split...")
    model.eval()
    predictions = []
    references = []
    
    for i, item in enumerate(dataset[split]):
        # Process audio
        input_features = processor(
            item["audio"]["array"],
            sampling_rate=item["audio"]["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="en",  # Specify English
                task="transcribe"  # Specify transcription task
            )[0]
        
        # Decode prediction
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        predictions.append(transcription)
        references.append(item["text"])
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset[split])} examples")
    
    # Compute metrics
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Print some example predictions
    print("\nExample predictions:")
    for pred, ref in list(zip(predictions, references))[:3]:
        print(f"\nReference: {ref}")
        print(f"Prediction: {pred}")
    
    print(f"\nWER: {wer:.4f}")
    return wer

if __name__ == "__main__":
    print("Starting Whisper fine-tuning experiment...")
    
    # Authenticate with Hugging Face
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("Please set the HF_TOKEN environment variable")
    login(token)
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    print(f"Dataset loaded. Keys: {dataset.keys()}")
    
    # Create train/test split
    print("Creating train/test split...")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    print(f"Split created. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Load model and processor
    print(f"\nLoading model and processor: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Model and processor loaded successfully")
    
    # Evaluate baseline performance
    print("\nEvaluating baseline model...")
    baseline_wer = evaluate_model(model, dataset, processor)
    print(f"Baseline WER: {baseline_wer:.4f}")
    
    # Prepare dataset
    print("\nPreparing dataset for training...")
    
    # Process train split
    print("Processing train split...")
    processed_train = []
    for idx, example in enumerate(tqdm(dataset["train"])):
        processed_train.append(prepare_dataset(example, idx))
    
    # Process test split
    print("Processing test split...")
    processed_test = []
    for idx, example in enumerate(tqdm(dataset["test"])):
        processed_test.append(prepare_dataset(example, idx))
    
    # Create new dataset
    from datasets import Dataset
    processed_dataset = {}
    processed_dataset["train"] = Dataset.from_list(processed_train)
    processed_dataset["test"] = Dataset.from_list(processed_test)
    dataset = processed_dataset
    
    print("Dataset preparation complete")
    
    # Prepare training arguments
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=1000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        generation_max_length=225,
        predict_with_generate=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Fine-tune the model
    print("\nStarting fine-tuning...")
    trainer.train()
    
    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    finetuned_wer = evaluate_model(model, dataset, processor)
    
    # Save results
    print("\nSaving results...")
    with open("experiment_results.txt", "w") as f:
        f.write(f"Baseline WER: {baseline_wer:.4f}\n")
        f.write(f"Fine-tuned WER: {finetuned_wer:.4f}\n")
    
    print("\nExperiment complete!")
    print(f"Baseline WER: {baseline_wer:.4f}")
    print(f"Fine-tuned WER: {finetuned_wer:.4f}") 