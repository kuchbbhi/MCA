# -------------------------
# Lexical Ambiguity 
# --------------------------

# 1. Rule Based (Lesk Algorithm) 
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Downloads needed for Lesk + tokenization
# nltk.download('punkt')
# nltk.download('punkt_tab') # This is needed in newer versions of nltk. If not used we get a warning. 
# nltk.download('wordnet')
# nltk.download('stopwords')

STOPWORD = set(stopwords.words('english'))

def lesk(word, sentence):
    words = nltk.word_tokenize(sentence.lower())    
    best_sense, max_overlap = None, 0
    
    for synonym in wordnet.synsets(word): # Synonym sets contains all possible meanings of a word
        '''we are creating a list of words that define the meaning of the synonym.
            This includes :
              1. definition
              2. lemma names
              3. examples
              
            Then we make all words lowercase and remove stopwords and non-alphabetic words.
            '''
        meaning_words = synonym.definition().split() + synonym.lemma_names() + synonym.examples()
        
        final_meaning_words = []
        for w in meaning_words:
            if w.isalpha() and w.lower() not in STOPWORD:
                final_meaning_words.append(w.lower())
        
        overlap = len(set(final_meaning_words) & set(words))

        if overlap > max_overlap:
            best_sense = synonym
            max_overlap = overlap
    return best_sense

sent1 = "He went to the bank to deposit money."
sent2 = "They sat on the bank of the river."

print("Sentence 1:", lesk("bank", sent1))
print("Sentence 2:", lesk("bank", sent2))


# -----------------------
# 2. Statistical (Naive Bayes) 
# !pip install scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Inputs
sentences = ["I deposited money in the bank", 
                 "She took a loan from the bank", 
                 "They sat on the river bank", 
                 "The river bank was flooded"]
labels = ["finance", "finance", "river", "river"]


vectorizer = CountVectorizer(stop_words='english')

sentence_vector = vectorizer.fit_transform(sentences)

model = MultinomialNB().fit(sentence_vector, labels)

print(model.predict(vectorizer.transform(["deposit at bank"])))
print(model.predict(vectorizer.transform(["walked by river bank"])))


# -------------------------------
## Syntactic Ambiguity
# -------------------------------

##### 1. Rule Based (CFG - Context Free Grammer)
from nltk import CFG, ChartParser

# We are defining our own Grammar Rules
grammar = CFG.fromstring("""
# Sentence
S -> NP VP

# Noun Phrase
NP -> 'I'
NP -> 'the' N
NP -> 'the' N PP

# Verb Phrase
VP -> V NP
VP -> VP PP

# Prepositional Phrase
PP -> P NP

# Nouns
N -> 'man'
N -> 'telescope'

# Verbs
V -> 'saw'

# Prepositions
P -> 'with'
""")


# Sentence to parse
sentence = "I saw the man with the telescope".split()

# Parse using rules
parser = ChartParser(grammar)

for tree in parser.parse(sentence):
    tree.pretty_print()



##### 2. Statistical (PCFG - Probabilistic Context Free Grammer)

from nltk import PCFG, ViterbiParser

# We are defining our own Grammar Rules
probabilistic_grammar = PCFG.fromstring("""
# Sentence
S -> NP VP [1.0]

# Noun Phrase
NP -> 'I' [0.3]
NP -> 'the' N [0.4]
NP -> 'the' N PP [0.3]

# Verb Phrase
VP -> V NP [0.4]
VP -> VP PP [0.6]

# Prepositional Phrase
PP -> P NP [1.0]

# Nouns
N -> 'man' [0.5]
N -> 'telescope' [0.5]

# Verbs
V -> 'saw' [1.0]

# Prepositions
P -> 'with' [1.0]
""")


# Sentence to parse
sentence = "I saw the man with the telescope".split()

# Parse using Probability Rules
parser = ViterbiParser(probabilistic_grammar)

for tree in parser.parse(sentence):
    tree.pretty_print()



# -------------------------------------
## Semantic Ambiguity
# -------------------------------------

##### 1. Rule Based (Using POS Tagging)

import nltk

# nltk.download('averaged_perceptron_tagger_eng')
sentence = "Visiting relatives can be annoying."

def semantic_rule(sentence):
    words = nltk.word_tokenize(sentence)          
    pos = nltk.pos_tag(words)

    # If first word is a verb (V), then "visiting" is an action
    if pos[0][1].startswith('V'):                 
        return "Meaning: The act of visiting relatives is annoying."
    else:
        return "Meaning: Relatives who are visiting are annoying."

print("Rule-Based:", semantic_rule(sentence))


##### 2. Statistical (Cue Based Method)

import nltk

# Clue word lists for meaning detection
cue_activity = {"annoying" , "going", "travel", "trip", "activity"}
cue_people = {"relatives", "family", "friend", "guest", "visitor"}


def cue_disambiguate(sentence):
    words = set(w.lower() for w in nltk.word_tokenize(sentence))

    score_activity = len(words & cue_activity)  # Common words with activity-related words
    score_people = len(words & cue_people)      # Common words with people-related words

    if score_activity > score_people:
        return "Activity sense (the act of visiting)."
    elif score_people > score_activity:
        return "People sense (relatives who visit)."
    else:
        return "Not sure / ambiguous"

print("Cue Method 1:", cue_disambiguate("Visiting relatives can be annoying."))
print("Cue Method 2:", cue_disambiguate("Relatives visiting the city can be noisy."))