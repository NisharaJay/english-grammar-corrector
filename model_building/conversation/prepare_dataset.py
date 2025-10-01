import os
import pandas as pd
from datasets import load_dataset

# ----------------------------
# Load DailyDialog dataset
# ----------------------------
dataset = load_dataset("daily_dialog", trust_remote_code=True)


# ----------------------------
# Format conversation
# ----------------------------
def format_dailydialog_conversation(dialog):
    """
    Convert a list of utterances into 'User:' / 'Bot:' style text
    """
    formatted = []
    for i, utt in enumerate(dialog):
        speaker = "User" if i % 2 == 0 else "Bot"
        formatted.append(f"{speaker}: {utt}")
    return "\n".join(formatted)

def dataset_to_dataframe(split):
    dialogs = dataset[split]["dialog"]
    formatted_dialogs = [format_dailydialog_conversation(d) for d in dialogs]
    return pd.DataFrame({"dialog": formatted_dialogs})

# ----------------------------
# Save as CSVs
# ----------------------------
os.makedirs("./dataset/conversation-dataset", exist_ok=True)

dataset_to_dataframe("train").to_csv("./dataset/conversation/train.csv", index=False)
dataset_to_dataframe("validation").to_csv("./dataset/conversation/validation.csv", index=False)
dataset_to_dataframe("test").to_csv("./dataset/conversation/test.csv", index=False)

print("✅ DailyDialog converted to CSVs with User/Bot format!")
