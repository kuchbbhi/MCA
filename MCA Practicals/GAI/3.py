# A) PROMPT DEBUGGING
# !pip install sacrebleu rouge-score sentence-transformers transformers

import itertools
import re
from typing import List, Dict
from transformers import pipeline

# ------------------------------
# Language Model Mapping
# ------------------------------
LANG_MODELS = {
    "Marathi": "Helsinki-NLP/opus-mt-en-mr",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Bengali": "shhossain/opus-mt-en-to-bn",
}
# Preload translation pipelines
TRANSLATORS = {
    lang: pipeline("translation", model=model_name)
    for lang, model_name in LANG_MODELS.items()
}

# ------------------------------
# Generate prompt variations
# ------------------------------
def generate_variations(base_prompt: str, variations: Dict[str, List[str]]) -> List[str]:
    keys = list(variations.keys())
    combos = list(itertools.product(*(variations[k] for k in keys)))

    prompts = []
    for combo in combos:
        p = base_prompt
        for k, v in zip(keys, combo):
            p = p.replace("{" + k + "}", v)
        prompts.append(p)
    return prompts

# ------------------------------
# Run HuggingFace translation model
# ------------------------------
def hf_runner(prompt: str) -> str:
    # Detect language
    lang = None
    for L in LANG_MODELS:
        if L.lower() in prompt.lower():
            lang = L
            break

    if not lang:
        return "Could not detect target language."

    # Extract text inside single quotes
    match = re.search(r"'(.*?)'", prompt)
    sentence = match.group(1) if match else prompt

    translator = TRANSLATORS.get(lang)
    if translator is None:
        return f"No translator loaded for {lang}."

    try:
        return translator(sentence)[0]["translation_text"]
    except Exception as e:
        return f"Translation error: {e}"

# ------------------------------
# Run tests for all prompts
# ------------------------------
def run_prompt_tests(prompts: List[str]):
    results = []
    for p in prompts:
        resp = hf_runner(p)
        results.append({"prompt": p, "response": resp})
    return results

# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    base = "Translate into {tone} {language}: '{sentence}'"

    vars = {
        "tone": ["formal", "casual"],
        "language": ["Marathi", "Hindi", "Bengali"],
        "sentence": ["I am from Pune", "Sit on the chair"]
    }

    prompts = generate_variations(base, vars)
    results = run_prompt_tests(prompts)

    for r in results:
        print("Prompt:", r["prompt"])
        print("Response:", r["response"])
        print("---")


# B) PROMPT EVALUATION METRICS

# !pip install sacrebleu rouge-score sentence-transformers

import sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# Reference and candidate outputs
reference = ["The cat is on the mat"]
candidate = "The cat is on mat"

# ------------------------------
# 1. BLEU SCORE
# ------------------------------
bleu = sacrebleu.corpus_bleu([candidate], [reference])
print("BLEU:", bleu.score)

# ------------------------------
# 2. ROUGE SCORE
# ------------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(reference[0], candidate)

print("ROUGE-1:", scores["rouge1"].fmeasure)
print("ROUGE-L:", scores["rougeL"].fmeasure)

# ------------------------------
# 3. SEMANTIC SIMILARITY (Sentence Transformer)
# ------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

ref_emb = model.encode(reference[0], convert_to_tensor=True)
cand_emb = model.encode(candidate, convert_to_tensor=True)

similarity = util.cos_sim(ref_emb, cand_emb)
print("Cosine Similarity:", float(similarity))
