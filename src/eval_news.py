import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. Setup & Loading
# ==========================================
base_model_id = "mistralai/Mistral-7B-v0.1"
adapter_path = "./mistral_laptop_absa_full_final"
# Pointing to your NEW newspaper dataset for OOD testing
test_data_path = "media_lens_train.jsonl" 

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)

print("Loading Base Model (16-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Attaching Trained LoRA Adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ==========================================
# 2. Parsing the Test Data
# ==========================================
print(f"Loading News OOD data from {test_data_path}...")
with open(test_data_path, "r", encoding="utf-8") as f:
    test_lines = f.readlines()

y_true = []
y_pred = []

print(f"Starting inference on {len(test_lines)} news examples...")

# ==========================================
# 3. The Evaluation Loop
# ==========================================
for line in tqdm(test_lines):
    data = json.loads(line)
    full_text = data["text"]

    parts = full_text.split("[/INST]")
    prompt = parts[0] + "[/INST] "

    # Clean the true label
    true_sentiment = parts[1].replace("</s>", "").strip().lower()
    y_true.append(true_sentiment)

    # Generate prediction
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.1, do_sample=True)

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_prediction = full_response.split("[/INST]")[-1].strip().lower()

    if "positive" in raw_prediction:
        y_pred.append("positive")
    elif "negative" in raw_prediction:
        y_pred.append("negative")
    elif "neutral" in raw_prediction:
        y_pred.append("neutral")
    else:
        y_pred.append("invalid")

# ==========================================
# 4. OOD Metrics Output
# ==========================================
print("\n" + "="*60)
print("🧠 MISTRAL OOD TEST - INDIA MEDIA LENS (NEWSPAPERS)")
print("="*60)

acc = accuracy_score(y_true, y_pred)
print(f"\nOOD NEWS ACCURACY: {acc * 100:.2f}%\n")

print("CLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(y_true, y_pred, zero_division=0))

print("\nCONFUSION MATRIX:")
print("-" * 60)
labels = ["positive", "negative", "neutral"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

print(f"{'':<12} | {'Pred: Pos':<10} | {'Pred: Neg':<10} | {'Pred: Neu':<10}")
print("-" * 55)
for i, label in enumerate(labels):
    print(f"True: {label:<6} | {cm[i][0]:<10} | {cm[i][1]:<10} | {cm[i][2]:<10}")
print("="*60)
