### Tokenization

import nltk
# nltk.download('punkt')

text = "Google acquired DeepMind in London for $500 million."

# Word Tokenization
print(nltk.word_tokenize(text))

# Sentence Tokenization
print(nltk.sent_tokenize(text))


### POS Tagging

import nltk
# nltk.download('averaged_perceptron_tagger')

words = nltk.word_tokenize(text)
print(nltk.pos_tag(words))



### Named Entity Recognition

import spacy

# Load spaCy model
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)
for entity in doc.ents:
    print(entity.text, "->", entity.label_)