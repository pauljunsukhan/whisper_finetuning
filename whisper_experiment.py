#!/usr/bin/env python3
"""
Minimal Whisper fine-tuning script for throat microphone data.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
import gc
from difflib import SequenceMatcher

# Constants
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "pauljunsukhan/throatmic_codered"
OUTPUT_DIR = "whisper_finetuned"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract the input_features from the examples
        input_features = [feature["input_features"] for feature in features]

        # Convert to tensors if they aren't already
        input_features = [
            feat if isinstance(feat, torch.Tensor) else torch.tensor(feat, dtype=torch.float32) 
            for feat in input_features
        ]

        # Ensure consistent shape by padding
        # Whisper features are (num_mel_frames, feature_dim=80)
        max_length = max(feat.shape[0] for feat in input_features)

        padded_features = []
        for feat in input_features:
            if feat.shape[0] < max_length:
                # Pad with zeros
                padding = torch.zeros((max_length - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
                feat = torch.cat([feat, padding], dim=0)
            padded_features.append(feat)

        # Now we can safely stack
        batch = {"input_features": torch.stack(padded_features)}

        # Handle labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, idx, processor):
    try:
        # Process audio
        audio = batch["audio"]
        
        # Process audio without autocast - let Trainer handle precision
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        # Process text into labels directly
        batch["labels"] = processor.tokenizer(batch["text"], truncation=True).input_ids
        batch["input_features"] = input_features.cpu()  # Keep as float32 CPU tensor
        
        return batch
    except Exception as e:
        print(f"Error processing example {idx}: {str(e)}")
        return None

def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER and get detailed error stats
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    # Compute CER
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # Detailed error analysis
    error_stats = {
        'total_words': 0,
        'total_chars': 0,
        'substitutions': 0,
        'deletions': 0,
        'insertions': 0,
        'long_word_errors': 0,  # errors in words > 5 chars
        'short_word_errors': 0  # errors in words <= 5 chars
    }

    for ref, hyp in zip(label_str, pred_str):
        # Word-level analysis
        ref_words = ref.split()
        hyp_words = hyp.split()
        error_stats['total_words'] += len(ref_words)
        error_stats['total_chars'] += len(ref)

        # Use SequenceMatcher for detailed error analysis
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                error_stats['substitutions'] += max(i2-i1, j2-j1)
                # Analyze word lengths in errors
                for word in ref_words[i1:i2]:
                    if len(word) > 5:
                        error_stats['long_word_errors'] += 1
                    else:
                        error_stats['short_word_errors'] += 1
            elif tag == 'delete':
                error_stats['deletions'] += (i2-i1)
            elif tag == 'insert':
                error_stats['insertions'] += (j2-j1)

    # Calculate error rates and distributions
    total_words = max(1, error_stats['total_words'])  # Prevent division by zero
    metrics = {
        "wer": wer,
        "cer": cer,
        "num_samples": len(pred_str),
        "avg_words_per_sample": error_stats['total_words'] / len(pred_str),
        "avg_chars_per_sample": error_stats['total_chars'] / len(pred_str),
        "substitution_rate": error_stats['substitutions'] / total_words,
        "deletion_rate": error_stats['deletions'] / total_words,
        "insertion_rate": error_stats['insertions'] / total_words,
        "long_word_error_rate": error_stats['long_word_errors'] / total_words,
        "short_word_error_rate": error_stats['short_word_errors'] / total_words
    }

    # Log detailed examples with error analysis
    if len(pred_str) > 0:
        print("\nDetailed Error Analysis Examples:")
        for i in range(min(3, len(pred_str))):
            print(f"\nExample {i+1}:")
            print(f"Reference: {label_str[i]}")
            print(f"Prediction: {pred_str[i]}")
            
            # Analyze this specific example
            ref_words = label_str[i].split()
            hyp_words = pred_str[i].split()
            matcher = SequenceMatcher(None, ref_words, hyp_words)
            
            print("Error Breakdown:")
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != 'equal':
                    print(f"- {tag.upper()}: '{' '.join(ref_words[i1:i2])}' -> '{' '.join(hyp_words[j1:j2])}'")

    return metrics

if __name__ == "__main__":
    try:
        print("Starting Whisper fine-tuning experiment...")
        
        # Load dataset and split
        print("Loading dataset...")
        dataset = load_dataset(DATASET_NAME)
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        print(f"Split created. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
        
        # Load model and processor
        print("\nLoading model and processor...")
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        
        # Load model without specifying dtype - let Trainer handle precision
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            use_cache=False,  # Required for gradient checkpointing
            device_map="auto"  # Automatically handle device placement
        )
        model.gradient_checkpointing_enable()
        
        # Process datasets
        print("\nPreparing dataset for training...")
        
        # Process train split
        print("Processing train split...")
        processed_train = []
        skipped_train = 0
        for idx, example in enumerate(tqdm(dataset["train"])):
            processed = prepare_dataset(example, idx, processor)
            if processed is not None:
                processed_train.append(processed)
            else:
                skipped_train += 1
        
        # Process test split
        print("Processing test split...")
        processed_test = []
        skipped_test = 0
        for idx, example in enumerate(tqdm(dataset["test"])):
            processed = prepare_dataset(example, idx, processor)
            if processed is not None:
                processed_test.append(processed)
            else:
                skipped_test += 1
        
        print(f"\nSkipped samples: Train: {skipped_train}, Test: {skipped_test}")
        
        # Create new dataset
        processed_dataset = {
            "train": Dataset.from_list(processed_train),
            "test": Dataset.from_list(processed_test)
        }
        dataset = processed_dataset
        
        print("Dataset preparation complete")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            warmup_steps=50,
            max_steps=1000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=50,
            logging_steps=10,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            remove_unused_columns=False,
            generation_max_length=225,
            predict_with_generate=True,
            optim="adamw_hf",
            max_grad_norm=1.0,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            group_by_length=False,  # Disabled - not appropriate for audio features
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
            compute_metrics=lambda pred: compute_metrics(pred, processor),
            tokenizer=processor.tokenizer,  # Fixed: Use tokenizer instead of feature_extractor
        )
        
        # Evaluate baseline model
        print("\nEvaluating baseline model...")
        baseline_metrics = trainer.evaluate()
        print(f"\nBaseline WER: {baseline_metrics['eval_wer']:.4f}")
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Final evaluation
        print("\nEvaluating fine-tuned model...")
        final_metrics = trainer.evaluate()
        print(f"\nFinal WER: {final_metrics['eval_wer']:.4f}")
        
        # Save results
        print("\nSaving results...")
        with open("experiment_results.txt", "w") as f:
            f.write("=== Whisper Fine-tuning Results ===\n\n")
            
            def write_metrics(f, title, metrics):
                f.write(f"{title}:\n")
                f.write(f"Primary Metrics:\n")
                f.write(f"- WER: {metrics['eval_wer']:.4f}\n")
                f.write(f"- CER: {metrics['eval_cer']:.4f}\n")
                f.write(f"\nError Analysis:\n")
                f.write(f"- Substitution Rate: {metrics['eval_substitution_rate']:.4f}\n")
                f.write(f"- Deletion Rate: {metrics['eval_deletion_rate']:.4f}\n")
                f.write(f"- Insertion Rate: {metrics['eval_insertion_rate']:.4f}\n")
                f.write(f"- Long Word Error Rate: {metrics['eval_long_word_error_rate']:.4f}\n")
                f.write(f"- Short Word Error Rate: {metrics['eval_short_word_error_rate']:.4f}\n")
                f.write(f"\nStatistics:\n")
                f.write(f"- Average Words/Sample: {metrics['eval_avg_words_per_sample']:.2f}\n")
                f.write(f"- Average Chars/Sample: {metrics['eval_avg_chars_per_sample']:.2f}\n")
                f.write(f"- Samples Evaluated: {metrics['eval_num_samples']}\n\n")
            
            write_metrics(f, "Baseline Metrics", baseline_metrics)
            write_metrics(f, "Final Metrics", final_metrics)
            
            # Add improvement analysis with zero-WER handling
            f.write("\nImprovement Analysis:\n")
            wer_improvement = baseline_metrics['eval_wer'] - final_metrics['eval_wer']
            cer_improvement = baseline_metrics['eval_cer'] - final_metrics['eval_cer']
            
            # Safe calculation of improvement percentages
            if baseline_metrics['eval_wer'] > 0:
                wer_pct = (wer_improvement / baseline_metrics['eval_wer']) * 100
                f.write(f"- WER Improvement: {wer_improvement:.4f} ({wer_pct:.1f}%)\n")
            else:
                f.write(f"- WER Improvement: {wer_improvement:.4f} (baseline WER was 0)\n")
                
            if baseline_metrics['eval_cer'] > 0:
                cer_pct = (cer_improvement / baseline_metrics['eval_cer']) * 100
                f.write(f"- CER Improvement: {cer_improvement:.4f} ({cer_pct:.1f}%)\n")
            else:
                f.write(f"- CER Improvement: {cer_improvement:.4f} (baseline CER was 0)\n")
            
            # Add recommendations based on error analysis
            f.write("\nRecommendations:\n")
            if final_metrics['eval_long_word_error_rate'] > final_metrics['eval_short_word_error_rate']:
                f.write("- Focus on improving long word recognition\n")
            if final_metrics['eval_deletion_rate'] > final_metrics['eval_insertion_rate']:
                f.write("- Model tends to skip words, consider adjusting decoder parameters\n")
            if final_metrics['eval_substitution_rate'] > max(final_metrics['eval_deletion_rate'], final_metrics['eval_insertion_rate']):
                f.write("- High substitution rate suggests potential acoustic modeling issues\n")
        
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()