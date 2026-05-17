import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# --- LOAD DATA ---
train = pd.read_csv('edu_train.csv')
test  = pd.read_csv('edu_test.csv')
dev   = pd.read_csv('edu_dev.csv')

df = pd.concat([train, test, dev], ignore_index=True)
df = df[['source_article', 'updated_label']].copy()
df.columns = ['text', 'fallacy_type']
df = df.dropna()

print(f"Loaded {len(df)} rows")

# --- CUSTOM TAXONOMY ---
taxonomy_map = {
    'ad hominem':             'Relevance Fallacy',
    'appeal to emotion':      'Relevance Fallacy',
    'fallacy of relevance':   'Relevance Fallacy',
    'ad populum':             'Relevance Fallacy',
    'fallacy of credibility': 'Relevance Fallacy',

    'faulty generalization':  'Presumption Fallacy',
    'false causality':        'Presumption Fallacy',
    'false dilemma':          'Presumption Fallacy',
    'circular reasoning':     'Presumption Fallacy',

    'fallacy of extension':   'Clarity Fallacy',
    'fallacy of logic':       'Clarity Fallacy',
    'equivocation':           'Clarity Fallacy',
    'intentional':            'Clarity Fallacy',
}

df['taxonomy_group'] = df['fallacy_type'].map(taxonomy_map)

print("Custom taxonomy applied")
print(f"\nTaxonomy distribution:\n{df['taxonomy_group'].value_counts()}")

# --- TEXT CLEANING ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("\nCleaning text (takes ~30 seconds)...")
df['clean_text'] = df['text'].apply(clean_text)
print("Text cleaned")

# --- SAVE ---
df.to_csv('fallacy_clean.csv', index=False)
print("Saved fallacy_clean.csv")

# --- CHART 1: Fallacy types ---
plt.figure(figsize=(10, 6))
counts = df['fallacy_type'].value_counts()
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))
plt.barh(counts.index, counts.values, color=colors)
plt.xlabel('Number of Examples')
plt.title('Fallacy Type Distribution in Dataset')
plt.tight_layout()
plt.savefig('chart_fallacy_types.png', dpi=150)
plt.close()
print("Saved chart_fallacy_types.png")

# --- CHART 2: Taxonomy groups ---
plt.figure(figsize=(7, 5))
tax_counts = df['taxonomy_group'].value_counts()
colors2 = ['#2196F3', '#4CAF50', '#FF9800']
plt.bar(tax_counts.index, tax_counts.values, color=colors2, edgecolor='white', linewidth=1.2)
plt.ylabel('Number of Examples')
plt.title('Custom Taxonomy Group Distribution')
plt.tight_layout()
plt.savefig('chart_taxonomy_groups.png', dpi=150)
plt.close()
print("Saved chart_taxonomy_groups.png")

print("\nPhase 2 complete. Check your folder for 2 PNG charts.")
print(f"\nFinal dataset shape: {df.shape}")
print(df[['fallacy_type', 'taxonomy_group', 'clean_text']].head(3))