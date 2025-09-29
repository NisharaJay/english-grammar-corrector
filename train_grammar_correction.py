from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq
)
import torch
import numpy as np
import evaluate
import sacrebleu


#Load dataset
ds = load_dataset("dim/grammarly_coedit")

MODELNAME = "t5-small"
PREFIX = "grammar: "

tokenizer = T5Tokenizer.from_pretrained(MODELNAME)
model = T5ForConditionalGeneration.from_pretrained(MODELNAME)

# Clean dataset
def fix(example):
    string_list = example["src"].split(":")
    text = " ".join(string_list[1:]).strip()
    example["src"] = text
    return example

ds["train"] = ds["train"].map(fix)

dataset = ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]

# Preprocess
MAX_LEN = 64

def preprocess(example):
    input_text = PREFIX + example["src"]
    target_text = example["tgt"]

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=MAX_LEN
    )
    labels = tokenizer(
        target_text,
        truncation=True,
        max_length=MAX_LEN
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_ds.map(preprocess, batched=False)
tokenized_val = val_ds.map(preprocess, batched=False)

# Metrics
exact_match = evaluate.load("exact_match")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    em = exact_match.compute(predictions=decoded_preds, references=decoded_labels)["exact_match"]
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score

    return {"exact_match": em, "bleu": bleu}

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-grammar-corrector",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,   # larger batch size (try 8 if OOM)
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)

# Data collator (dynamic padding)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

model.save_pretrained("./t5-grammar-corrector")
tokenizer.save_pretrained("./t5-grammar-corrector")
trainer.save_model("./t5-grammar-corrector") 

eval_results = trainer.evaluate()
print("📊 Evaluation Results:", eval_results)

model = T5ForConditionalGeneration.from_pretrained("./t5-grammar-corrector")
tokenizer = T5Tokenizer.from_pretrained("./t5-grammar-corrector")

def correct_grammar(text: str):
    input_text = f"grammar: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    # Move to GPU if available
    input_ids = input_ids.to(model.device)

    output_ids = model.generate(input_ids, max_length=128, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Try on a sample sentence
test_sentence = "He go to school every day."
corrected = correct_grammar(test_sentence)
print(f"\n Original:  {test_sentence}")
print(f"Corrected: {corrected}")