# !pip install sacrebleu rouge_score sentence_transformers

# Theory: Prompt Debugging
# Prompt debugging involves generating variations of prompts and testing their outputs. This helps in identifying the most effective prompt for a given task.

# - `generate_variations`: Creates all possible combinations of prompt templates.
# - `mock_runner`: Simulates a model response (replace with actual model call for real use).
# - `run_prompt_tests`: Runs all prompt variations and collects responses.
# The following code cell demonstrates prompt debugging.

from typing import List, Dict
import itertools

# 1. Generate prompt variations
def generate_variations(
    base_prompt: str, variations: Dict[str, List[str]]
) -> List[str]:
    keys = list(variations.keys())
    combos = list(itertools.product(*(variations[k] for k in keys)))
    prompts = []
    for combo in combos:
        p = base_prompt
        for k, v in zip(keys, combo):
            p = p.replace("{" + k + "}", v)
        prompts.append(p)
    return prompts


# 2. Mock runner (replace with OpenAI/HF runner)
def mock_runner(prompt: str) -> str:
    return "Response to: " + prompt

# 3. Run tests
def run_prompt_tests(prompts: List[str]):
    results = []
    for p in prompts:
        resp = mock_runner(p)
        results.append({"prompt": p, "response": resp})
    return results

# Example Usage 1
base = "Translate into {tone} English: '{sentence}'"
vars = {"tone": ["formal", "casual"], "sentence": ["Bonjour", "Comment ça va?"]}

prompts = generate_variations(base, vars)
results = run_prompt_tests(prompts)

for r in results:
    print("Prompt:", r["prompt"])
    print("Response:", r["response"])
    print("---")
    
    
# Example Usage 2 (with different sentences)
base2 = "Summarize in {style} style: '{text}'"
vars2 = {
    "style": ["bullet", "paragraph"],
    "text": ["The sun rises in the east.", "Water is essential for life."],
}

prompts2 = generate_variations(base2, vars2)
results2 = run_prompt_tests(prompts2)

for r in results2:
    print("Prompt:", r["prompt"])
    print("Response:", r["response"])
    print("---")
    
# Theory: Performance Evaluation Metrics

# Performance metrics help evaluate the quality of generated text compared to reference outputs.

# - **Accuracy**: Measures exact match between prediction and reference.
# - **BLEU**: Evaluates n-gram overlap (commonly used for translation).
# - **ROUGE**: Measures overlap of sequences (used for summarization).
# - **Semantic Similarity**: Measures meaning similarity using embeddings.

# The following code cell defines functions for each metric and demonstrates their usage.


from typing import List
import sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


# 1. Accuracy / Exact Match
def accuracy(preds: List[str], refs: List[str]) -> float:
    return sum([p.strip() == r.strip() for p, r in zip(preds, refs)]) / len(preds)


# 2. BLEU Score
def bleu(preds: List[str], refs: List[str]) -> float:
    return sacrebleu.corpus_bleu(preds, [refs]).score


# 3. ROUGE Scores
def rouge(preds: List[str], refs: List[str]):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return scores


# 4. Semantic Similarity
def semantic_similarity(preds: List[str], refs: List[str]) -> List[float]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = model.encode(preds, convert_to_tensor=True)
    ref_emb = model.encode(refs, convert_to_tensor=True)
    sims = util.cos_sim(pred_emb, ref_emb)
    return [float(sims[i, i]) for i in range(len(preds))]


# Example Usage
preds = ["Hello", "How are you?"]
refs = ["Hello", "How are you?"]

print("Accuracy:", accuracy(preds, refs))
print("BLEU:", bleu(preds, refs))
print("ROUGE:", rouge(preds, refs))
print("Semantic similarity:", semantic_similarity(preds, refs))


## Metric Evaluation

# Reference and candidate outputs
reference = ["The cat is on the mat"]
candidate = "The cat is on mat"

# BLEU Score
bleu = sacrebleu.corpus_bleu([candidate], [reference])
print("BLEU:", bleu.score)

# ROUGE Score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(reference[0], candidate)
print("ROUGE-1:", scores["rouge1"].fmeasure)
print("ROUGE-L:", scores["rougeL"].fmeasure)

# Semantic Similarity
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
ref_emb = model.encode(reference[0], convert_to_tensor=True)
cand_emb = model.encode(candidate, convert_to_tensor=True)
similarity = util.cos_sim(ref_emb, cand_emb)
print("Cosine Similarity:", float(similarity))