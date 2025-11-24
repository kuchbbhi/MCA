# Lab 1: Ambiguity Resolution in Natural Language Processing

# !pip install nltk scikit-learn

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# 1. Lexical Ambiguity Resolution

STOP = set(stopwords.words('english'))

def simple_lesk(target_word, sentence):
    sent_tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
    best_sense = None
    max_overlap = 0
    for syn in wn.synsets(target_word):
        signature = ' '.join([syn.definition()] + syn.examples() + list(syn.lemma_names()))
        sig_tokens = [w.lower() for w in nltk.word_tokenize(signature)]
        sig_tokens = [w for w in sig_tokens if w.isalpha() and w not in STOP]
        overlap = sum((Counter(sig_tokens) & Counter(sent_tokens)).values())
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = syn
    return best_sense

sent1 = "He went to the bank to deposit his paycheck."
sent2 = "They sat on the bank of the river to watch the sunset."

print("Lesk result 1:", simple_lesk("bank", sent1))
print("Lesk result 2:", simple_lesk("bank", sent2))

train_examples = [
    ("I deposited money at the bank", "finance"),
    ("She took a loan from the bank", "finance"),
    ("The bank closed early on Friday", "finance"),
    ("The river bank was full of wildflowers", "river"),
    ("He walked along the bank listening to the water", "river"),
    ("Cattle grazed on the bank after the rain", "river")
]

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform([t for t, _ in train_examples])
y_train = [lbl for _, lbl in train_examples]

clf = MultinomialNB().fit(X_train, y_train)

def predict_bank_sense(sentence):
    x = vectorizer.transform([sentence])
    pred = clf.predict(x)[0]
    return pred, dict(zip(clf.classes_, clf.predict_proba(x)[0]))

print("NB prediction 1:", predict_bank_sense(sent1))
print("NB prediction 2:", predict_bank_sense(sent2))


# 2. Syntactic Ambiguity Resolution


from nltk import CFG, ChartParser, PCFG, ViterbiParser

# --- Rule-based: CFG with multiple parses ---
grammar = CFG.fromstring("""
S -> NP VP
NP -> 'I' | 'the' N | 'the' N PP
N -> 'man' | 'telescope'
VP -> V NP | VP PP
V -> 'saw'
PP -> P NP
P -> 'with'
""")

parser = ChartParser(grammar)
sentence = "I saw the man with the telescope".split()

print("\nAll parses (rule-based CFG):")
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()

pcfg = PCFG.fromstring("""
S -> NP VP [1.0]
NP -> 'I' [0.2] | Det N [0.4] | Det N PP [0.4]
VP -> V NP [0.6] | VP PP [0.4]
Det -> 'the' [1.0]
N -> 'man' [0.5] | 'telescope' [0.5]
V -> 'saw' [1.0]
PP -> P NP [1.0]
P -> 'with' [1.0]
""")

viterbi_parser = ViterbiParser(pcfg)

print("\nMost probable parse (PCFG):")
for tree in viterbi_parser.parse(sentence):
    print(tree)
    tree.pretty_print()
    break


# 3. Semantic Ambiguity Resolution

sent = "Visiting relatives can be annoying."
tokens = nltk.word_tokenize(sent)
pos = nltk.pos_tag(tokens, tagset='universal')  # use universal tagset
print("\nPOS tags:", pos)

def disambiguate_visiting(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(tokens, tagset='universal')
    if pos[0][1] == 'VERB' and pos[1][1] == 'NOUN':
        return "Gerund reading: 'Visiting relatives' = the act of visiting is annoying."
    else:
        return "Participle reading: 'Visiting relatives' = relatives who are visiting."

print(disambiguate_visiting(sent))

cue_activity = {"visit", "going", "travel", "trip", "activity"}
cue_people = {"relative", "family", "friend", "guest", "visitor"}

def cue_disambiguate(sentence):
    words = set(w.lower() for w in nltk.word_tokenize(sentence) if w.isalpha())
    score_activity = len(words & cue_activity)
    score_people = len(words & cue_people)
    if score_activity > score_people:
        return "Activity sense (the action of visiting)."
    elif score_people > score_activity:
        return "People sense (relatives who are visiting)."
    else:
        return "Undecided by cues."

print("Cue method 1:", cue_disambiguate("Visiting relatives can be annoying."))
print("Cue method 2:", cue_disambiguate("Relatives visiting the city can be noisy."))
