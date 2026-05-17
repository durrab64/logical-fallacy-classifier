# Logical Fallacy Classification with Custom Taxonomy

**National University of Modern Languages, Lahore**  
4th Semester | Section B | Artificial Intelligence

## Team
- Muhammad Rehan
- Muhammad Durrab  
- Muhammad Ahmed
- Fatima Rana

## Project Overview
An AI system that detects and classifies logical fallacies from text using 
a fine-tuned BERT model with a custom 3-group taxonomy.

## Custom Taxonomy
| Group | Fallacy Types |
|---|---|
| Relevance Fallacy | Ad Hominem, Appeal to Emotion, Ad Populum, Fallacy of Relevance, Fallacy of Credibility |
| Presumption Fallacy | Faulty Generalization, False Causality, False Dilemma, Circular Reasoning |
| Clarity Fallacy | Fallacy of Extension, Fallacy of Logic, Equivocation, Intentional |

## Results
| Model | Accuracy | F1 Score |
|---|---|---|
| Baseline (TF-IDF + LR) | 45.9% | 39.4% |
| BERT (fine-tuned) | 56.3% | 46.5% |

## Dataset
Logic dataset by Jin et al., EMNLP 2022 — 2,449 labeled examples, 13 fallacy types.

## Tech Stack
- Python, PyTorch, Hugging Face Transformers
- Scikit-learn, NLTK, Pandas
- Flask, HTML/CSS/JS

## How to Run
1. Download the fine-tuned model from Google Drive (link below) and place 
   it in `bert_type_saved/`
2. Install dependencies: `pip install flask flask-cors transformers torch`
3. Run: `python app.py`
4. Open: `http://localhost:5000`

## Model Download
[Download bert_type_saved from Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)