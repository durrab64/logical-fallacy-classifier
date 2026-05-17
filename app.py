from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np, json, os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load model once at startup
MODEL_PATH = './bert_type_saved/bert_type_saved'
print("Loading model...")

with open(f'{MODEL_PATH}/label_classes.json') as f:
    classes = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model ready.")

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
    'fallacy of extension':   'Misrepresents or exaggerates an opponents argument.',
    'false dilemma':          'Presents only two options when more actually exist.',
    'fallacy of credibility': 'Accepts or rejects claim based solely on the sources reputation.',
    'equivocation':           'Uses the same word with different meanings to mislead.',
}

taxonomy_colors = {
    'Relevance Fallacy':   '#2E86C1',
    'Presumption Fallacy': '#1E8449',
    'Clarity Fallacy':     '#7D3C98',
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    fallacy    = classes[pred_idx]
    taxonomy   = taxonomy_map.get(fallacy, 'Unknown')

    # Top 3 predictions
    top3_idx  = np.argsort(probs)[::-1][:3]
    top3 = [
        {'fallacy': classes[i].title(), 'confidence': round(float(probs[i]) * 100, 1)}
        for i in top3_idx
    ]

    return jsonify({
        'fallacy':     fallacy.title(),
        'taxonomy':    taxonomy,
        'confidence':  round(confidence, 1),
        'description': descriptions.get(fallacy, ''),
        'color':       taxonomy_colors.get(taxonomy, '#2E86C1'),
        'top3':        top3,
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)