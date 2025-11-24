import nltk
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def textrank_summarizer(text, top_n=2):
    sentences = nltk.sent_tokenize(text)

    # Step 1: Convert sentences to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Step 2: Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 3: Build graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Step 4: Apply PageRank
    scores = nx.pagerank(graph)

    # Step 5: Rank and pick top sentences
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return " ".join([sent for _, sent in ranked[:top_n]])


article = """
Artificial Intelligence is transforming the world.
AI helps in healthcare and finance.
However, AI also raises ethical questions.
Experts suggest using AI responsibly.
"""

print("TEXTRANK SUMMARY:\n", textrank_summarizer(article, 2))
