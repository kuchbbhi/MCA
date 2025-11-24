import nltk
from nltk import bigrams, trigrams
from nltk.util import ngrams

# Download punkt tokenizer
# nltk.download('punkt')

text = """Natural Language Processing is fun and interesting. 
Language models help in predicting the next word in a sentence."""

# Tokenize
words = nltk.word_tokenize(text.lower())
print(words)


# Building Bigram Model
bigrams = {}
for i in range(len(words) - 1):
    w1, w2 = words[i], words[i+1]
    
    # This is to prevent error when w1 is not present, i.e. the first word of sentence
    if w1 not in bigrams:
        bigrams[w1] = []
    
    bigrams[w1].append(w2)

print("Bigram Model:", bigrams)


def predict_bigram(word):
    if word in bigrams:
        return bigrams[word][0]   # just return the first word for simplicity
    return "No prediction"

print("Next word after 'is':", predict_bigram("is"))
print("Next word after 'language':", predict_bigram("language"))



# Building Trigram Model
trigrams = {}
for i in range(len(words) - 2):
    key = (words[i], words[i+1])
    next_word = words[i+2]
    if key not in trigrams:
        trigrams[key] = []
    trigrams[key].append(next_word)

print("Trigram Model:", trigrams)



def predict_trigram(w1, w2):
    key = (w1, w2)
    if key in trigrams:
        return trigrams[key][0]
    return "No prediction"

print("Next word after ('language', 'processing'):", predict_trigram('language', 'processing'))
print("Next word after ('next', 'word'):", predict_trigram('next', 'word'))
