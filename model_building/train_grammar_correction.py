# evaluate_model.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from collections import Counter
import sacrebleu

# -------------------------------
# Config
# -------------------------------
MODEL_DIR = "./model_building/t5-grammar-corrector"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Token-level F1 function
# -------------------------------
def f1_token_level(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# -------------------------------
# Metrics computation
# -------------------------------
def compute_metrics(predictions, references):
    em_score = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    f1_scores = [f1_token_level(p, r) for p, r in zip(predictions, references)]
    f1_avg = sum(f1_scores) / len(f1_scores)
    
    return {
        "exact_match": em_score,
        "bleu": bleu_score,
        "f1": f1_avg
    }

# -------------------------------
# Load model & tokenizer
# -------------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)

# -------------------------------
# Load and prepare official test dataset
# -------------------------------
test_ds = load_dataset("dim/grammarly_coedit", split="test")

# Clean dataset
def fix(example):
    string_list = example["src"].split(":")
    text = " ".join(string_list[1:]).strip()
    example["src"] = text
    return example

test_ds = test_ds.map(fix)

# -------------------------------
# Generate predictions
# -------------------------------
preds = []
refs = []

print("Generating predictions on test set...")
for example in test_ds:
    input_text = "grammar: " + example["src"]
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    input_ids = input_ids.to(DEVICE)

    output_ids = model.generate(input_ids, max_length=128, num_beams=4)
    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    preds.append(pred_text)
    refs.append(example["tgt"])

# -------------------------------
# Compute metrics
# -------------------------------
metrics = compute_metrics(preds, refs)
print("📊 Test Set Evaluation Results:")
print(metrics)

# -------------------------------
# Optional: show some sample predictions
# -------------------------------
print("\nSample predictions:")
for i in range(5):
    print(f"Original : {test_ds[i]['src']}")
    print(f"Target   : {refs[i]}")
    print(f"Predicted: {preds[i]}\n")
