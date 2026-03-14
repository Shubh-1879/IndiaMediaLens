import pandas as pd
import json

# ==========================================
# 1. Configuration
# ==========================================
input_csv = "newspaper_data.csv"
output_jsonl = "media_lens_train.jsonl"

print(f"Reading {input_csv}...")
df = pd.read_csv(input_csv)

def create_jsonl_line(row):
    # Combine the three columns into one aspect term
    aspect_parts = [str(row[col]).strip() for col in ['central body', 'political figure', 'affiliate'] 
                    if pd.notna(row[col]) and str(row[col]).strip() != ""]
    aspect_term = " ".join(aspect_parts)
    
    text = str(row['Text']).strip()
    
    # Map the labels from your News CSV to the Mistral-trained labels
    raw_stance = str(row['Stance']).strip().lower()
    if raw_stance == "pro":
        stance = "positive"
    elif raw_stance == "anti":
        stance = "negative"
    else:
        stance = "neutral"
    
    # The prompt remains exactly as it was for the Laptop data
    instruction = (
        f"<s>[INST] Identify the polarity towards the aspect term '{aspect_term}' "
        f"in the following sentence. Output only the polarity (positive, negative, or neutral).\n"
        f"Sentence: {text} [/INST] {stance}</s>"
    )
    
    return {"text": instruction}

# ==========================================
# 2. Processing & Saving
# ==========================================
print("Converting rows to Mistral-instruction format...")
with open(output_jsonl, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json_record = create_jsonl_line(row)
        f.write(json.dumps(json_record) + "\n")

print(f"Successfully saved {len(df)} rows to {output_jsonl}")