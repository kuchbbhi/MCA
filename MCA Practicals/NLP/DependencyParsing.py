import spacy
nlp = spacy.load("en_core_web_sm")

def identify_roles(sentence):
    doc = nlp(sentence)
    roles = {"Agent": None, "Patient": None, "Instrument": None}

    for token in doc:
        # Agent = subject of verb
        if token.dep_ in ("nsubj", "nsubjpass"): # Nominal Subject and Passive Nominal Subject
            roles["Agent"] = token.text

        # Patient = object of verb
        elif token.dep_ in ("dobj", "obj"): # Direct Object and Object
            roles["Patient"] = token.text

        # Instrument = prepositional phrase with "with"
        elif token.dep_ == "pobj" and token.head.text == "with": # Prepositional Object
            roles["Instrument"] = token.text

    return roles

sentence =  "John cut the bread with a knife."
print("Roles:", identify_roles(sentence))
