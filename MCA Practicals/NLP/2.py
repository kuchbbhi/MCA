# Lab 2

import nltk, spacy
from transformers import pipeline

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")  # fixes the tokenization issue
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")  # NEW fix for POS tagging

# Load spaCy model
spacy.cli.download("en_core_web_sm")  # ensure model is available
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Google acquired DeepMind in London for $500 million."

# --- 1. Tokenization ---
tokens = nltk.word_tokenize(text)
print("Word Tokens:", tokens)

sent_tokens = nltk.sent_tokenize(text)
print("\nSentence Tokens:", sent_tokens)

# --- 2. POS Tagging ---
pos_tags = nltk.pos_tag(tokens)
print("\nPOS Tags:", pos_tags)

# --- 3. NER with spaCy ---
doc = nlp(text)
print("\nNER (spaCy):")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

# --- 4. NER with Transformer (BERT-based model) ---
ner_model = pipeline("ner", grouped_entities=True)
print("\nNER (Transformer):")
print(ner_model(text))
