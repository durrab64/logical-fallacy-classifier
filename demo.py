# demo.py — run this locally in VS Code after saving models from Colab
import gradio as gr

# Taxonomy map for display
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

descriptions = {
    'ad hominem':             'Attacks the person making the argument, not the argument itself.',
    'faulty generalization':  'Draws a broad conclusion from a small or unrepresentative sample.',
    'false causality':        'Assumes one event caused another just because it came first.',
    'ad populum':             'Claims something is true because many people believe it.',
    'circular reasoning':     'Uses the conclusion as a premise — the argument proves itself.',
    'appeal to emotion':      'Manipulates emotions instead of using logical reasoning.',
    'fallacy of relevance':   'Uses irrelevant information to distract from the real issue.',
    'fallacy of logic':       'Contains a structural error that makes the argument invalid.',
    'intentional':            'A deliberate deceptive argument meant to mislead.',
    'fallacy of extension':   'Misrepresents or exaggerates an opponent\'s argument.',
    'false dilemma':          'Presents only two options when more actually exist.',
    'fallacy of credibility': 'Accepts or rejects a claim based solely on the source\'s reputation.',
    'equivocation':           'Uses the same word with different meanings to mislead.',
}

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch, pandas as pd
import numpy as np

# Load label encoders from your cleaned data
df = pd.read_csv('fallacy_clean.csv').dropna(subset=['fallacy_type','taxonomy_group'])
le_type = LabelEncoder()
le_type.fit(df['fallacy_type'])

print("Loading model... (takes ~20 seconds)")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# We'll load the model you download from Colab
model = AutoModelForSequenceClassification.from_pretrained('./bert_type_saved')
model.eval()
print("Model ready.")

def predict(text):
    if not text.strip():
        return "Please enter an argument.", "", "", ""

    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    fallacy    = le_type.classes_[pred_idx]
    taxonomy   = taxonomy_map.get(fallacy, 'Unknown')
    description = descriptions.get(fallacy, '')

    return (
        f"{fallacy.title()}",
        f"{taxonomy}",
        f"{confidence:.1f}%",
        description
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter an argument here..."),
    outputs=[
        gr.Textbox(label="Detected Fallacy"),
        gr.Textbox(label="Taxonomy Group"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="What this means"),
    ],
    title="Logical Fallacy Classifier",
    description="Enter any argument and the AI will detect the logical fallacy and classify it into our custom taxonomy.",
    examples=[
        ["Everyone is buying this product so it must be great."],
        ["You can't trust his opinion on climate change, he's not a scientist."],
        ["If we allow gay marriage, next people will want to marry animals."],
        ["I took vitamin C and my cold went away, so vitamin C cured my cold."],
    ]
)

demo.launch()