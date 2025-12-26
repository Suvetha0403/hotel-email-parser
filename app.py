from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json, re

app = Flask(__name__)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_REPO = "suvetha04/llama3.2-lora-hotel-parser"

# Load model once
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model = PeftModel.from_pretrained(base_model, LORA_REPO)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

HTML = """
<!doctype html>
<title>Hotel Email Parser</title>
<h2>Hotel Email Parser</h2>
<form method="POST">
Subject:<br><input name="subject" size="100"><br><br>
Body:<br><textarea name="body" rows="12" cols="100"></textarea><br><br>
<input type="submit">
</form>
{% if result %}
<pre>{{ result }}</pre>
{% endif %}
"""

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group()) if match else {}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        prompt = f"""Extract hotel booking info as JSON.
Subject: {request.form['subject']}
Body: {request.form['body']}
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=300)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        result = json.dumps(extract_json(text), indent=2)
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
