# Lab 3: Implement a morphological parser for English words

import re
# 1. Irregular Lexicon (Exception Handling)
IRREGULAR_NOUNS = {
    "child": "children",
    "mouse": "mice",
    "man": "men",
    "woman": "women",
    "person": "people",
    "foot": "feet",
    "tooth": "teeth",
    "goose": "geese",
}
IRREGULAR_VERBS_PAST = {
    "go": "went",
    "be": "was",
    "eat": "ate",
    "run": "ran",
    "come": "came",
    "have": "had",
    "make": "made",
    "see": "saw",
    "write": "wrote",
}
# 2. Morphological Generators (Lexical → Surface)
def pluralize(noun):
    """Generate plural form of English nouns."""
    if noun in IRREGULAR_NOUNS:
        return IRREGULAR_NOUNS[noun]
    if re.search(r"(s|x|z|ch|sh)$", noun):
        return noun + "es"             
    if re.search(r"[^aeiou]y$", noun):
        return noun[:-1] + "ies"       
    return noun + "s"                
    
def past_tense(verb):
    """Generate past tense of verbs."""
    if verb in IRREGULAR_VERBS_PAST:
        return IRREGULAR_VERBS_PAST[verb]
    if verb.endswith("e"):
        return verb + "d"              
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ied"       
    if re.search(r"[aeiou][bcdfghjklmnpqrstvwxyz]$", verb):
        return verb + verb[-1] + "ed"  
    return verb + "ed"                
    
def present_participle(verb):
    """Generate present participle (-ing) form."""
    if verb.endswith("ie"):
        return verb[:-2] + "ying"      
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"      
    if re.search(r"[aeiou][bcdfghjklmnpqrstvwxyz]$", verb):
        return verb + verb[-1] + "ing" 
    return verb + "ing"   
# 3. Morphological Analyzers (Surface → Lexical + Features)

def analyze_plural(word):
    """Analyze plural nouns."""
    for base, plural in IRREGULAR_NOUNS.items():
        if word == plural:
            return f"{base}+N+PL"
    if re.search(r"ies$", word):
        return f"{word[:-3]}y+N+PL"
    if re.search(r"(es)$", word):
        return f"{word[:-2]}+N+PL"
    if word.endswith("s"):
        return f"{word[:-1]}+N+PL"
    return "UNKNOWN"
def analyze_past(word):
    """Analyze past tense verbs."""
    for base, past in IRREGULAR_VERBS_PAST.items():
        if word == past:
            return f"{base}+V+PAST"
    if re.search(r"ied$", word):
        return f"{word[:-3]}y+V+PAST"
    if re.search(r"([bcdfghjklmnpqrstvwxyz])\\1ed$", word):
        return f"{word[:-3]}+V+PAST"
    if word.endswith("ed"):
        return f"{word[:-2]}+V+PAST"
    return "UNKNOWN"
def analyze_ing(word):
    """Analyze present participle verbs."""
    if word.endswith("ying"):
        return f"{word[:-4]}ie+V+PROG"
    if re.search(r"([bcdfghjklmnpqrstvwxyz])\\1ing$", word):
        return f"{word[:-4]}+V+PROG"
    if word.endswith("ing"):
        return f"{word[:-3]}+V+PROG"
    return "UNKNOWN"
# 4. High-Level Wrapper
def generate(lemma, pos, feature):
    """Generate word form given lemma, POS, and morphological feature."""
    if pos == "N" and feature == "PL":
        return pluralize(lemma)
    if pos == "V" and feature == "PAST":
        return past_tense(lemma)
    if pos == "V" and feature == "PROG":
        return present_participle(lemma)
    return lemma
def analyze(word):
    """Return possible analyses for the word."""
    analyses = []
    plural = analyze_plural(word)
    if plural != "UNKNOWN":
        analyses.append(plural)
    past = analyze_past(word)
    if past != "UNKNOWN":
        analyses.append(past)
    ing = analyze_ing(word)
    if ing != "UNKNOWN":
        analyses.append(ing)
    return analyses or ["UNKNOWN"]
# 5. Demonstration / Test
if __name__ == "__main__":
    print("=== GENERATION (Lexical → Surface) ===")
    print("cat + N + PL  →", generate("cat", "N", "PL"))
    print("city + N + PL →", generate("city", "N", "PL"))
    print("try + V + PAST →", generate("try", "V", "PAST"))
    print("run + V + PROG →", generate("run", "V", "PROG"))
    print("go + V + PAST →", generate("go", "V", "PAST"))
    print("make + V + PROG →", generate("make", "V", "PROG"))
    print("\n=== ANALYSIS (Surface → Lexical + Features) ===")
    words = ["cats", "cities", "stopped", "running", "making", "went", "children"]
    for w in words:
        print(f"{w:10} → {analyze(w)}")