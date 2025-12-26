import os
import json
import re
import torch

from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Environment Variables
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # MUST be set in Render

# -----------------------------
# Model Configuration
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_REPO = "suvetha04/llama3.2-lora-hotel-parser"

# -----------------------------
# Load Model ONCE (IMPORTANT)
# -----------------------------
print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    LORA_REPO,
    token=HF_TOKEN
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🚀 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("🚀 Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_REPO,
    token=HF_TOKEN
)

model.eval()
print("✅ Model loaded successfully")

# -----------------------------
# HTML UI
# -----------------------------
HTML_TEMPLATE = """
<!doctype html>
<title>Hotel Email Parser</title>

<h2>🏨 Hotel Email Parser</h2>

<form method="POST">
  <label><b>Subject:</b></label><br>
  <input type="text" name="subject" size="100" required><br><br>

  <label><b>Email Body:</b></label><br>
  <textarea name="body" rows="12" cols="100" required></textarea><br><br>

  <input type="submit" value="Process Email"
         style="font-size:16px;padding:10px 20px;">
</form>

{% if result %}
<hr>
<h3>📄 Extracted JSON</h3>
<pre>{{ result }}</pre>
{% endif %}
"""

# -----------------------------
# JSON Extractor
# -----------------------------
def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        return {"error": str(e)}
    return {}

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        subject = request.form.get("subject", "")
        body = request.form.get("body", "")

        prompt = f"""
### Instruction:
Classify this hotel-related email and extract details as JSON.

### Input:
Subject: {subject}

Body:
{body}

### Output:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        parsed = extract_json(text)
        result = json.dumps(parsed, indent=2)

    return render_template_string(HTML_TEMPLATE, result=result)

# -----------------------------
# Health Check
# -----------------------------
@app.route("/health")
def health():
    return "OK"

# -----------------------------
# Run App (Render Compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
