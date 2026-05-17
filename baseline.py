import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# --- LOAD CLEANED DATA ---
df = pd.read_csv('fallacy_clean.csv')
df = df.dropna(subset=['clean_text', 'fallacy_type', 'taxonomy_group'])
df['clean_text'] = df['clean_text'].astype(str)
print(f"Loaded {len(df)} rows")

# ═══════════════════════════════════════════
# MODEL A — Classify specific fallacy type
# ═══════════════════════════════════════════
print("\n--- MODEL A: Fallacy Type Classifier ---")

X = df['clean_text']
y_type = df['fallacy_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_type, test_size=0.2, random_state=42, stratify=y_type)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

model_type = LogisticRegression(max_iter=1000, random_state=42)
model_type.fit(X_train_vec, y_train)
y_pred_type = model_type.predict(X_test_vec)

acc_type = accuracy_score(y_test, y_pred_type)
f1_type  = f1_score(y_test, y_pred_type, average='macro')

print(f"Accuracy : {acc_type:.4f} ({acc_type*100:.1f}%)")
print(f"F1 Score : {f1_type:.4f} ({f1_type*100:.1f}%)")
print(f"\nDetailed Report:\n{classification_report(y_test, y_pred_type)}")

# Confusion matrix — Model A
cm_a = confusion_matrix(y_test, y_pred_type, labels=model_type.classes_)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(cm_a, display_labels=model_type.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues', xticks_rotation=45)
ax.set_title('Baseline Model A — Fallacy Type Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_baseline_type.png', dpi=150)
plt.close()
print("Saved cm_baseline_type.png")

# ═══════════════════════════════════════════
# MODEL B — Classify taxonomy group
# ═══════════════════════════════════════════
print("\n--- MODEL B: Taxonomy Group Classifier ---")

y_tax = df['taxonomy_group']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_tax, test_size=0.2, random_state=42, stratify=y_tax)

tfidf2 = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec2 = tfidf2.fit_transform(X_train2)
X_test_vec2  = tfidf2.transform(X_test2)

model_tax = LogisticRegression(max_iter=1000, random_state=42)
model_tax.fit(X_train_vec2, y_train2)
y_pred_tax = model_tax.predict(X_test_vec2)

acc_tax = accuracy_score(y_test2, y_pred_tax)
f1_tax  = f1_score(y_test2, y_pred_tax, average='macro')

print(f"Accuracy : {acc_tax:.4f} ({acc_tax*100:.1f}%)")
print(f"F1 Score : {f1_tax:.4f} ({f1_tax*100:.1f}%)")
print(f"\nDetailed Report:\n{classification_report(y_test2, y_pred_tax)}")

# Confusion matrix — Model B
cm_b = confusion_matrix(y_test2, y_pred_tax, labels=model_tax.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp2 = ConfusionMatrixDisplay(cm_b, display_labels=model_tax.classes_)
disp2.plot(ax=ax, colorbar=False, cmap='Greens', xticks_rotation=15)
ax.set_title('Baseline Model B — Taxonomy Group Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_baseline_taxonomy.png', dpi=150)
plt.close()
print("Saved cm_baseline_taxonomy.png")

# --- SUMMARY ---
print("\n" + "="*45)
print("BASELINE RESULTS SUMMARY")
print("="*45)
print(f"Fallacy Type  — Accuracy: {acc_type*100:.1f}%  F1: {f1_type*100:.1f}%")
print(f"Taxonomy Group — Accuracy: {acc_tax*100:.1f}%  F1: {f1_tax*100:.1f}%")
print("="*45)
print("\nPhase 3 complete. 2 confusion matrix PNGs saved.")