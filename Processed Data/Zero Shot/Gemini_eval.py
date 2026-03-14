import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file_path = "Zero Shot Result.xlsx" 

print(f"Loading data from {file_path}...")
df = pd.read_excel(file_path)

# ==========================================
# Data Cleaning & Normalization
# ==========================================
def clean_label(text):
    text = str(text).lower()
    if "positive" in text: return "positive"
    if "negative" in text: return "negative"
    if "neutral" in text: return "neutral"
    if "conflict" in text: return "conflict"
    return "invalid"

# Apply the strict cleaner to BOTH columns
df['Clean_True'] = df['True Polarity'].apply(clean_label)
df['Clean_Pred'] = df['Gemini Polarity'].apply(clean_label)

# Count and drop the corrupted rows so they don't ruin the math
corrupted_rows = len(df[df['Clean_True'] == 'invalid'])
if corrupted_rows > 0:
    print(f"⚠️ WARNING: Found and removed {corrupted_rows} rows with corrupted True Polarity data.")

df_clean = df[df['Clean_True'] != 'invalid']

y_true = df_clean['Clean_True']
y_pred = df_clean['Clean_Pred']

# ==========================================
# Final Advanced Metrics
# ==========================================
print("\n" + "="*60)
print("🧠 GEMINI ZERO/FEW-SHOT - FINAL EVALUATION METRICS")
print("="*60)

acc = accuracy_score(y_true, y_pred)
print(f"\nOVERALL ACCURACY: {acc * 100:.2f}%\n")

print("CLASSIFICATION REPORT:")
print("-" * 60)
# We specify the exact labels we care about so it ignores hallucinations
target_labels = ["positive", "negative", "neutral", "conflict"]
print(classification_report(y_true, y_pred, labels=target_labels, zero_division=0))

print("\nCONFUSION MATRIX:")
print("Rows = True | Columns = Pred")
print("-" * 60)
# We only plot the main 3 for the matrix to keep it clean, or you can add conflict
matrix_labels = ["positive", "negative", "neutral"]
cm = confusion_matrix(y_true, y_pred, labels=matrix_labels)

print(f"{'':<14} | {'Pred: Pos':<10} | {'Pred: Neg':<10} | {'Pred: Neu':<10}")
print("-" * 55)
for i, label in enumerate(matrix_labels):
    print(f"True: {label:<8} | {cm[i][0]:<10} | {cm[i][1]:<10} | {cm[i][2]:<10}")
print("="*60)