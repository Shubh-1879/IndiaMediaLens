import pandas as pd
import json

# --- CONFIGURATION ---
INPUT_FILE = "Batch 1 to fine tune.xlsx"   # Rename to your file
OUTPUT_FILE = "train.jsonl"

# Your exact columns
COL_SENTENCE = "sentence"
COL_ASPECT = "term" 
COL_POLARITY = "Human Annotated Polarity"


def format_mistral_instruction(row):
    sentence = str(row[COL_SENTENCE]).strip()
    aspect = str(row[COL_ASPECT]).strip()
    polarity = str(row[COL_POLARITY]).strip()

    instruction = f"Identify the sentiment towards the aspect '{aspect}' in the following sentence. Output only the sentiment (positive, negative, or neutral)."
    user_input = f"Sentence: {sentence}"
    
    # Mistral Format
    text = f"<s>[INST] {instruction}\n{user_input} [/INST] {polarity}</s>"
    return {"text": text}

print(f"Reading {INPUT_FILE}...")
df = pd.read_excel(INPUT_FILE)

# Write with newline='\n' to force Linux-style line endings
with open(OUTPUT_FILE, "w", encoding="utf-8", newline='\n') as f:
    for _, row in df.iterrows():
        example = format_mistral_instruction(row)
        f.write(json.dumps(example) + "\n")

print(f"Success! Upload '{OUTPUT_FILE}' to the HPC.")