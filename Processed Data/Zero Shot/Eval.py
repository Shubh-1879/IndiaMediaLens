import pandas as pd
from sklearn.metrics import classification_report

# 1. Load your final joined dataset
df = pd.read_excel(r"C:\Users\shubh\Downloads\MSc 3rd Sem\IndiaMediaLens\Processed Data\Zero Shot\Zero Shot Result.xlsx")

# 2. Define your expected valid classes
valid_classes = ['positive', 'negative', 'neutral']

# 3. PREPROCESSING & EDGE CASE HANDLING
# Convert everything to lowercase and strip accidental spaces (Case-Insensitive)
df['True Polarity'] = df['True Polarity'].astype(str).str.strip().str.lower()
df['Gemini Polarity'] = df['Gemini Polarity'].astype(str).str.strip().str.lower()

# Edge Case A: Gemini missed the prediction (Null/NaN from the SQL Left Join)
# Pandas converts NaN to the string 'nan' when we use .astype(str)
df['Gemini Polarity'] = df['Gemini Polarity'].replace('nan', 'unpredicted')

# Edge Case B: Dirty Ground Truth
# If your original file had a typo (e.g., 'positiv'), it breaks the math.
# We flag these so you know they exist, but filter them out for the final score.
invalid_truth_mask = ~df['True Polarity'].isin(valid_classes)
if invalid_truth_mask.sum() > 0:
    print(f"⚠️ WARNING: Found {invalid_truth_mask.sum()} rows with invalid 'True Polarity'.")
    print("These rows are excluded from the final metrics.\n")
    df_clean = df[~invalid_truth_mask].copy()
else:
    df_clean = df.copy()

# 4. ROW-BY-ROW ERROR ANALYSIS (Adding columns to your CSV)
# This flags exactly what happened for easy manual review
def flag_error_type(row):
    truth = row['True Polarity']
    pred = row['Gemini Polarity']
    
    if pred == truth:
        return 'Correct Match'
    elif pred == 'unpredicted':
        return 'Missing Prediction'
    elif pred not in valid_classes:
        return f'Hallucinated Label ({pred})'
    else:
        return 'Incorrect Class'

df_clean['Prediction_Status'] = df_clean.apply(flag_error_type, axis=1)

# Save the tagged file for your own manual review
df_clean.to_csv("Laptop_Error_Analysis.csv", index=False)

# 5. GENERATE FINAL METRICS
y_true = df_clean['True Polarity']
y_pred = df_clean['Gemini Polarity']

print("--- ABSA Classification Report ---")
# labels=valid_classes forces sklearn to only report on your target labels
# zero_division=0 prevents crashes if Gemini completely failed to predict a certain class
report = classification_report(
    y_true, 
    y_pred, 
    labels=valid_classes, 
    zero_division=0
)
print(report)