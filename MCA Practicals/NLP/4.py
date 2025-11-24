import nltk          # NLP toolkit for tokenization
import re            # Regex for word extraction
from nltk.util import ngrams   # To generate unigrams, bigrams, trigrams
from collections import Counter # Efficient word frequency counter
import math          # For log, exp in perplexity calculation
import matplotlib.pyplot as plt # For plotting frequency distributions


# TOKENIZATION FUNCTION
try:
    nltk.download("punkt", quiet=True)      # Punkt tokenizer (pre-trained)
    nltk.download("punkt_tab", quiet=True)  # Needed in some NLTK versions
    from nltk import word_tokenize

    def tokenize(text):
        return word_tokenize(text.lower())

except:
    # If NLTK fails, use regex: \b\w+\b matches full words only.
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())
    
# 
text = """The dog barks. The cat meows. The dog runs fast.
The cat sleeps. The dog eats food. The cat drinks milk."""

tokens = tokenize(text)

unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))


unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)


def predict_next_bigram(word):
    # Collect candidate words that follow the given word
    candidates = {w2: count for (w1, w2), count in bigram_freq.items() if w1 == word}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)

def predict_next_trigram(w1, w2):
    candidates = {w3: count for (x1, x2, w3), count in trigram_freq.items() if x1 == w1 and x2 == w2}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


# PERPLEXITY CALCULATION

def perplexity(test_tokens, model="bigram"):
    N = len(test_tokens)  # total words
    log_prob = 0

    if model == "bigram":
        for i in range(1, N):
            w1, w2 = test_tokens[i-1], test_tokens[i]
            prob = (bigram_freq[(w1, w2)] + 1) / (unigram_freq[(w1,)] + len(unigram_freq))
            log_prob += math.log(prob)

    elif model == "trigram":
        for i in range(2, N):
            w1, w2, w3 = test_tokens[i-2], test_tokens[i-1], test_tokens[i]
            prob = (trigram_freq[(w1, w2, w3)] + 1) / (bigram_freq[(w1, w2)] + len(unigram_freq))
            log_prob += math.log(prob)

    return math.exp(-log_prob / N)


# VISUALIZATION
def plot_top_ngrams(freq_dict, title, n=10):
    # Convert tuple keys like ('the','dog') → "the dog"
    ngrams_list = [" ".join(gram) for gram, _ in freq_dict.most_common(n)]
    counts = [count for _, count in freq_dict.most_common(n)]

    # Plot bar chart
    plt.figure(figsize=(8,4))
    plt.bar(ngrams_list, counts, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("N-grams")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()
    

# OUTPUT
print("Tokens:", tokens)  # Show tokenized words
print("Bigram Prediction after 'dog':", predict_next_bigram("dog"))  # Predict next word after "dog"
print("Trigram Prediction after 'the dog':", predict_next_trigram("the", "dog"))  # Predict after "the dog"

print("Bigram Perplexity:", perplexity(tokens, model="bigram"))  # Evaluate model performance
print("Trigram Perplexity:", perplexity(tokens, model="trigram"))


# 
plot_top_ngrams(unigram_freq, "Top Unigrams", n=10)

# Top 10 most frequent word pairs
plot_top_ngrams(bigram_freq, "Top Bigrams", n=10)

# Top 10 most frequent word triples
plot_top_ngrams(trigram_freq, "Top Trigrams", n=10)