# Lab 7 : Implement LEXRANK or TextRank to generate summaries from news articles.

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")  

import nltk
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("punkt")
nltk.download("punkt_tab")

def textrank_summarize(text, top_n=3):
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([sent for _, sent in ranked_sentences[:top_n]])
    return summary

article = """
Artificial Intelligence (AI) is transforming industries worldwide.
From healthcare to finance, AI applications are becoming increasingly common.
In healthcare, AI helps doctors diagnose diseases more accurately and quickly.
In finance, AI algorithms detect fraud and assist in investment decisions.
Despite its benefits, AI raises ethical concerns, including job displacement and privacy issues.
Experts suggest that careful regulation and ethical guidelines are needed to ensure AI’s positive impact on society.
"""

summary = textrank_summarize(article, top_n=2)
print("Original Article:\n", article)
print("\nGenerated Summary:\n", summary)