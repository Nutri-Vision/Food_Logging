import spacy
from pathlib import Path

def load_trained_model(model_path="models/food_ner"):
    p = Path(model_path)
    if p.exists() and p.is_dir():
        try:
            return spacy.load(model_path)
        except Exception as e:
            print("Failed to load spaCy model:", e)
    # Return None when model unavailable
    return None
