import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./model/distilgpt2_grammar"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.eval()

conversation_history = []

print("💬 English Practice Bot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Keep conversation as a list
    conversation_history.append(f"User: {user_input}")

    # Construct prompt for model
    prompt = "\n".join(conversation_history) + "\nBot:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,    # only generate new tokens for Bot
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Take only the new bot response
    bot_reply = decoded[len(prompt):].strip()
    print("Bot:", bot_reply)

    # Add bot reply to conversation
    conversation_history.append(f"Bot: {bot_reply}")
