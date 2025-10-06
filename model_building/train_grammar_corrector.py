import warnings
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import sacrebleu


# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Config
MODELNAME = "t5-small"       # Changed from t5-base to t5-small
PREFIX = "grammar: "
MAX_LEN = 64

# Load model + tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODELNAME)
model = T5ForConditionalGeneration.from_pretrained(MODELNAME)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()


# Load & clean dataset
ds = load_dataset("dim/grammarly_coedit")

def fix(example):
    string_list = example["src"].split(":")
    text = " ".join(string_list[1:]).strip()
    example["src"] = text
    return example

ds["train"] = ds["train"].map(fix)
ds["train"] = ds["train"].shuffle(seed=42).select(range(20000))

dataset = ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]


# Preprocessing
def preprocess(example):
    input_text = PREFIX + example["src"]
    target_text = example["tgt"]
    model_inputs = tokenizer(input_text, truncation=True, max_length=MAX_LEN)
    labels = tokenizer(target_text, truncation=True, max_length=MAX_LEN)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_ds.map(preprocess, batched=False)
tokenized_val = val_ds.map(preprocess, batched=False)


# Metrics
exact_match_metric = evaluate.load("exact_match")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    em = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)["exact_match"]
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
    return {"exact_match": em, "bleu": bleu}


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-grammar-corrector-small",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    fp16_full_eval=True,
    logging_dir="./logs",
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# Train + Save
trainer.train()
model.save_pretrained("./t5-grammar-corrector-small")
tokenizer.save_pretrained("./t5-grammar-corrector-small")
trainer.save_model("./t5-grammar-corrector-small")


# Evaluate
eval_results = trainer.evaluate()
print("📊 Evaluation Results:", eval_results)


# Grammar correction function
model = T5ForConditionalGeneration.from_pretrained("./t5-grammar-corrector-small")
tokenizer = T5Tokenizer.from_pretrained("./t5-grammar-corrector-small")

def correct_grammar(text: str):
    input_text = f"{PREFIX}{text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    input_ids = input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=64, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test sample sentence
test_sentence = "He go to school every day."
corrected = correct_grammar(test_sentence)
print(f"\nOriginal: {test_sentence}")
print(f"Corrected: {corrected}")
