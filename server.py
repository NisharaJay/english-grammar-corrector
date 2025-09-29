import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
from concurrent.futures import ThreadPoolExecutor


MODEL_DIR = "./t5-grammar-corrector"  # path to your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

#Load Model & Tokenizer
print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

#Explicitly set decoder_start_token_id for T5
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.pad_token_id

if hasattr(model.config, 'bos_token_id') and model.config.bos_token_id is None:
    model.config.bos_token_id = tokenizer.pad_token_id

print(f"Model loaded on device: {DEVICE}")
print(f"Decoder start token ID: {model.config.decoder_start_token_id}")
print(f"BOS token ID: {getattr(model.config, 'bos_token_id', 'Not set')}")

#Evaluation Metrics
exact_match = evaluate.load("exact_match")
f1_metric = evaluate.load("f1")
bleu_metric = evaluate.load("sacrebleu")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#ThreadPool for async GPU calls
executor = ThreadPoolExecutor(max_workers=1)

# WebSocket Endpoint
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")
            reference = payload.get("reference")  # optional: expected correct sentence

            if not message:
                await websocket.send_text(json.dumps({"reply": "Please send a valid message."}))
                continue

            input_text = f"correct grammar: {message}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                max_length=MAX_LENGTH
            ).to(DEVICE)

            # Run model.generate asynchronously
            def generate_reply():
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_length=MAX_LENGTH,
                        num_beams=4,
                        early_stopping=True,
                        decoder_start_token_id=model.config.decoder_start_token_id
                    )
                return tokenizer.decode(output_ids[0], skip_special_tokens=True)

            reply = await asyncio.get_event_loop().run_in_executor(executor, generate_reply)

            # Prepare response
            result = {"reply": reply}

            # Compute metrics if reference is provided
            if reference:
                pred_tokens = reply.split()
                ref_tokens = reference.split()
                f1_score = f1_metric.compute(predictions=[pred_tokens], references=[ref_tokens])["f1"]
                em_score = exact_match.compute(predictions=[reply], references=[reference])["exact_match"]
                bleu_score = bleu_metric.compute(predictions=[reply], references=[[reference]])["score"]

                result.update({
                    "f1": f1_score,
                    "exact_match": em_score,
                    "bleu": bleu_score
                })

            await websocket.send_text(json.dumps(result))
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()

# Health Check
@app.get("/")
async def root():
    return {"status": "online", "model": "T5 Grammar Corrector"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(DEVICE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)