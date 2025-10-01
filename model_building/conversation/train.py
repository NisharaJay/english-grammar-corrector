import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import evaluate

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "distilgpt2"   # smaller, faster GPT-2
MODEL_DIR = "./model/distilgpt2_grammar"
MAX_LENGTH = 128            # context length
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
EPOCHS = 2                  # increase if you have time/GPU
LR = 5e-5

# ----------------------------
# Load dataset (CSV files from prepare_dataset.py)
# ----------------------------
dataset = load_dataset(
    "csv",
    data_files={
        "train": "./dataset/conversation/train.csv",
        "validation": "./dataset/conversation/validation.csv",
        "test": "./dataset/conversation/test.csv"
    }
)

# ----------------------------
# Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_function(examples):
    texts = [str(d) for d in examples["dialog"]]  # already User/Bot formatted
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ----------------------------
# Model
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

# ----------------------------
# Data collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------
# Training args
# ----------------------------
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    weight_decay=0.01,
    logging_dir=f"{MODEL_DIR}/logs",
    logging_steps=100,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    eval_accumulation_steps=1,
    push_to_hub=False
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ----------------------------
# Train
# ----------------------------
trainer.train()

# ----------------------------
# Save final model
# ----------------------------
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# ----------------------------
# Evaluation on test set
# ----------------------------
print("🔎 Running evaluation on test set...")

bleu = evaluate.load("sacrebleu")

test_texts = dataset["test"]["dialog"]
preds, refs = [], []

for text in test_texts[:50]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # instead of max_length=MAX_LENGTH
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    preds.append(pred)
    refs.append(text)

# Exact Match
exact_match = np.mean([p.strip() == r.strip() for p, r in zip(preds, refs)])

# BLEU
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]

# F1 (character-level for simplicity)
def f1_char_level(pred, ref):
    pred_chars, ref_chars = list(pred), list(ref)
    common = set(pred_chars) & set(ref_chars)
    if not common:
        return 0.0
    tp = sum(min(pred_chars.count(c), ref_chars.count(c)) for c in common)
    precision = tp / len(pred_chars) if pred_chars else 0
    recall = tp / len(ref_chars) if ref_chars else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

f1_scores = [f1_char_level(p, r) for p, r in zip(preds, refs)]
f1 = np.mean(f1_scores)

print("📊 Test Set Metrics:")
print(f"  Exact Match: {exact_match:.4f}")
print(f"  BLEU: {bleu_score:.2f}")
print(f"  F1: {f1:.4f}")
