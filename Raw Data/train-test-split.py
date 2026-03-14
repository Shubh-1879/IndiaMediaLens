import pandas as pd
from sklearn.model_selection import train_test_split
import json

# 1. Point to your exact local Windows file path
file_path = r"C:\Users\shubh\Downloads\MSc 3rd Sem\IndiaMediaLens\Raw Data\Laptop_True_Data.csv"

print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Split the data: 85% for training, 15% for testing
# random_state=42 ensures you get the exact same split if you run it again
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

print(f"Split complete: {len(train_df)} training rows, {len(test_df)} testing rows.")

# 3. The formatting function
def convert_to_mistral_jsonl(dataframe, output_filename):
    with open(output_filename, "w", encoding="utf-8") as outfile:
        for index, row in dataframe.iterrows():
            
            # --- IMPORTANT: Change these if your CSV column headers are different ---
            sentence = str(row['Sentence']).strip()
            aspect = str(row['Aspect Term']).strip()
            sentiment = str(row['True Polarity']).strip().capitalize()
            # ------------------------------------------------------------------------
            
            # Apply the exact Mistral [INST] wrapper we validated
            formatted_text = (
                f"<s>[INST] Identify the polarity towards the aspect term '{aspect}' "
                f"in the following sentence. Output only the polarity (positive, negative, or neutral).\n"
                f"Sentence: {sentence} [/INST] {sentiment}</s>"
            )
            
            # Write to JSONL
            json_record = {"text": formatted_text}
            outfile.write(json.dumps(json_record) + "\n")

# 4. Generate the two files
convert_to_mistral_jsonl(train_df, "absa_train.jsonl")
convert_to_mistral_jsonl(test_df, "absa_test.jsonl")

print("Files generated successfully! Ready to upload to the HPC.")