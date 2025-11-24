"""
wsd_examples.py

This script demonstrates three Word Sense Disambiguation (WSD) methods:
1) simplified_lesk  — based on word overlap between context and WordNet glosses.
2) extended_lesk    — a weighted version of Lesk that includes related senses.
3) embed_lesk       — uses sentence embeddings to compute similarity instead of word overlap.

WSD = Word Sense Disambiguation:
--------------------------------
It is the process of identifying which sense (meaning) of a word is used in a sentence,
when the word has multiple meanings (polysemy).

For example:
 - "bank" in "He sat on the river bank" → 'river edge'
 - "bank" in "She went to the bank to deposit money" → 'financial institution'

The Lesk algorithm uses dictionary definitions (WordNet glosses) to disambiguate words
based on the overlap between the context and possible sense definitions.
"""

import math
import re
from collections import Counter
from typing import Optional, List, Tuple

# ====================================================================
# NLTK & WordNet Setup
# ====================================================================
try:
    import nltk
    from nltk.corpus import wordnet as wn  # WordNet lexical database
    from nltk.corpus import stopwords
    from nltk import word_tokenize

    # Download required data quietly
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    raise RuntimeError("NLTK and WordNet are required. Install nltk and run again.") from e

STOPWORDS = set(stopwords.words("english"))

# ====================================================================
# Text Preprocessing Utility
# ====================================================================
def tokenize_and_normalize(text: str) -> List[str]:
    """
    Convert input text into a list of lowercase tokens (words),
    remove stopwords and very short tokens.
    This normalizes text for comparison.
    """
    text = text.lower()
    # Use regex to extract words only
    tokens = re.findall(r"\b\w+\b", text)
    # Filter out stopwords and one-letter words
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# ====================================================================
# Simplified Lesk Algorithm
# ====================================================================
def simplified_lesk(context_sentence: str, ambiguous_word: str, pos: Optional[str]=None) -> Optional[wn.synset]:
    """
    Simplified Lesk Algorithm (Baseline version)
    ---------------------------------------------
    Theoretical idea:
      - For each possible sense (synset) of a word:
          • Take its definition (gloss) and example sentences.
          • Compute the overlap of words with the context sentence.
      - The sense with the highest overlap is chosen.

    Parameters:
        context_sentence: The sentence containing the ambiguous word.
        ambiguous_word: The target word whose sense we need to disambiguate.
        pos: Optional Part-of-Speech ('n', 'v', 'a', 'r').

    Returns:
        Best matching synset (WordNet sense).
    """
    context = set(tokenize_and_normalize(context_sentence))
    max_overlap = 0
    best_sense = None

    # Get all candidate meanings (synsets)
    candidates = wn.synsets(ambiguous_word, pos=pos) if pos else wn.synsets(ambiguous_word)
    if not candidates:
        return None

    for s in candidates:
        # Construct the “signature” for each synset:
        # Definition + Example sentences + Lemma names (synonyms)
        signature = set(tokenize_and_normalize(s.definition()))
        for ex in s.examples():
            signature.update(tokenize_and_normalize(ex))
        for lemma in s.lemma_names():
            signature.update(tokenize_and_normalize(lemma.replace("_", " ")))

        # Compute overlap (intersection) between context and signature
        overlap = len(context & signature)

        # Choose synset with maximum overlap (first tie-break: more common sense)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = s

    return best_sense

# ====================================================================
# Extended Lesk Algorithm (Weighted Overlap)
# ====================================================================
def extended_lesk(context_sentence: str, ambiguous_word: str, pos: Optional[str]=None) -> Optional[wn.synset]:
    """
    Extended Lesk Algorithm
    -----------------------
    Theory:
      - Improves upon the original Lesk by:
          1. Assigning weights to words from different sources.
          2. Including related senses like hypernyms (general terms).
          3. Using frequency-based tie-breakers.

      Weighted sources:
          • Definition words      → weight 2.0
          • Example sentence words→ weight 1.5
          • Lemma names (synonyms)→ weight 2.0
          • Hypernym info         → weight 0.8

    Returns:
        Synset with maximum weighted overlap.
    """
    context = set(tokenize_and_normalize(context_sentence))
    best = None
    best_score = -1.0

    candidates = wn.synsets(ambiguous_word, pos=pos) if pos else wn.synsets(ambiguous_word)

    for s in candidates:
        sig_counter = Counter()

        # Add words from definition (weight 2)
        def_tokens = tokenize_and_normalize(s.definition())
        for t in def_tokens:
            sig_counter[t] += 2.0

        # Add words from example sentences (weight 1.5)
        for ex in s.examples():
            for t in tokenize_and_normalize(ex):
                sig_counter[t] += 1.5

        # Add lemma (synonym) words (weight 2)
        for lemma in s.lemma_names():
            for t in tokenize_and_normalize(lemma.replace("_", " ")):
                sig_counter[t] += 2.0

        # Add hypernym (general sense) information with smaller weight (0.8)
        for hyper in s.hypernyms():
            for t in tokenize_and_normalize(hyper.definition()):
                sig_counter[t] += 0.8
            for lemma in hyper.lemma_names():
                for t in tokenize_and_normalize(lemma.replace("_", " ")):
                    sig_counter[t] += 0.8

        # Compute weighted overlap score
        score = sum(sig_counter[t] for t in context if t in sig_counter)

        # Small bonus for frequent senses (based on number of lemmas)
        score += 0.01 * len(s.lemma_names())

        # Select best scoring sense
        if score > best_score:
            best = s
            best_score = score

    return best

# ====================================================================
# Embedding-based Lesk (Semantic Similarity)
# ====================================================================
def embed_lesk(context_sentence: str, ambiguous_word: str, pos: Optional[str]=None):
    """
    Embedding-based Lesk Algorithm
    ------------------------------
    Theory:
      - Uses vector representations (embeddings) of text to measure similarity.
      - Each sense’s description (gloss + examples + synonyms) is converted into a vector.
      - The context sentence is also embedded.
      - The cosine similarity between context and each sense determines the best match.

    This approach captures semantic similarity even when words do not overlap exactly.

    Dependencies:
      pip install sentence-transformers

    Model used:
      all-MiniLM-L6-v2  → lightweight, fast, semantic sentence encoder
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        raise RuntimeError("sentence-transformers is not installed. Install with `pip install sentence-transformers`")

    # Load pretrained sentence embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    context = context_sentence

    # Retrieve candidate synsets
    candidates = wn.synsets(ambiguous_word, pos=pos) if pos else wn.synsets(ambiguous_word)
    if not candidates:
        return None

    # Encode context as an embedding
    ctx_emb = model.encode(context, convert_to_tensor=True)

    synset_texts = []
    synset_map = []

    # Prepare text for each synset (definition + examples + lemmas)
    for s in candidates:
        parts = [s.definition()]
        parts += s.examples()
        parts += [ln.replace("_", " ") for ln in s.lemma_names()]
        text = " . ".join(parts)
        synset_texts.append(text)
        synset_map.append(s)

    # Encode all synsets
    syn_embs = model.encode(synset_texts, convert_to_tensor=True)
    # Compute cosine similarity between context and each synset embedding
    cos_sims = util.cos_sim(ctx_emb, syn_embs)[0]
    best_idx = int(cos_sims.argmax())

    # Return synset with highest semantic similarity
    return synset_map[best_idx]

# ====================================================================
# Demo / Example Execution
# ====================================================================
if __name__ == "__main__":
    # Test sentences with ambiguous words
    tests = [
        ("I need to withdraw money from the bank to pay rent.", "bank"),
        ("The river bank was filled with green reeds.", "bank"),
        ("He watered the plant in the corner of the room.", "plant"),
        ("The factory is a large plant with many workers.", "plant"),
        ("He swung the bat and hit the ball.", "bat"),
        ("A bat hung upside down in the cave.", "bat"),
        ("She listens to rock all the time.", "rock"),
        ("We climbed up on the rock by the sea.", "rock"),
    ]

    # ===============================================================
    # 1) Simplified Lesk
    # ===============================================================
    print("\n==== Simplified Lesk results ====\n")
    for sent, word in tests:
        sense = simplified_lesk(sent, word)
        print(f"Context: {sent}")
        if sense:
            print(f" Word: {word} -> Synset: {sense.name()}  | Definition: {sense.definition()}\n")
        else:
            print(f" Word: {word} -> NO SENSE FOUND\n")

    # ===============================================================
    # 2) Extended Lesk
    # ===============================================================
    print("\n==== Extended Lesk results ====\n")
    for sent, word in tests:
        sense = extended_lesk(sent, word)
        print(f"Context: {sent}")
        if sense:
            print(f" Word: {word} -> Synset: {sense.name()}  | Definition: {sense.definition()}\n")
        else:
            print(f" Word: {word} -> NO SENSE FOUND\n")

    # ===============================================================
    # 3) Embedding-based Lesk (optional)
    # ===============================================================
    try:
        print("\n==== Embedding-based Lesk results (sentence-transformers) ====\n")
        for sent, word in tests:
            sense = embed_lesk(sent, word)
            print(f"Context: {sent}")
            if sense:
                print(f" Word: {word} -> Synset: {sense.name()}  | Definition: {sense.definition()}\n")
            else:
                print(f" Word: {word} -> NO SENSE FOUND\n")
    except RuntimeError as e:
        print("\nEmbedding-based Lesk skipped (sentence-transformers not installed).")
        print("To run it, install: pip install sentence-transformers\n")
