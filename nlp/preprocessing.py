import re
import spacy

nlp = spacy.blank("en")

def clean_text(text: str) -> str:
    """Basic text cleaning for food descriptions"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text