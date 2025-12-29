import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------------
# Hugging Face Token (optional)
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ HF_TOKEN not set, assuming public model access")

# -----------------------------
# Model paths
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_REPO = "suvetha04/llama3.2-lora-hotel-parser"

# -----------------------------
# Load tokenizer
# -----------------------------
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_auth_token=HF_TOKEN if HF_TOKEN else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 4-bit quantization config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 compute for efficiency
    bnb_4bit_use_double_quant=True
)

# -----------------------------
# Load base model in 4-bit
# -----------------------------
print("🔄 Loading base model in 4-bit (GPU recommended)...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",  # GPU preferred
        use_auth_token=HF_TOKEN if HF_TOKEN else None
    )
except Exception as e:
    print("❌ Failed to load 4-bit model:", e)
    raise RuntimeError("4-bit model may be too large for free CPU. GPU recommended.")

# -----------------------------
# Load LoRA adapter
# -----------------------------
print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("✅ Base model + LoRA adapter loaded successfully")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Hotel Email Parser API (4-bit)")

class EmailRequest(BaseModel):
    subject: str
    body: str

# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: EmailRequest):
    prompt = f"""
Extract booking details from this email and return JSON only.

Subject: {data.subject}
Email: {data.body}

JSON:
"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "subject": data.subject,
        "parsed_output": response
    }
