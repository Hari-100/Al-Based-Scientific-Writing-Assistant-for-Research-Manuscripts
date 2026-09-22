import re
import spacy

# no model download needed, sentencizer is rule based
_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")

# matches a whole parenthetical that contains a year, e.g. "(Smith, 2020; Jones, 2021)"
CITATION_GROUP_PATTERN = re.compile(r"\(([^()]*\d{4}[a-z]?[^()]*)\)")
NAME_PATTERN = re.compile(r"[A-Z][A-Za-z\-]+")


def split_sentences(text):
    # protect "et al." from being read as a sentence end, then restore it after
    guarded = text.replace("et al.", "et al@")
    doc = _nlp(guarded)
    sentences = [s.text.strip().replace("et al@", "et al.") for s in doc.sents if s.text.strip()]
    return sentences


def extract_citations(sentence):
    citations = []
    for group in CITATION_GROUP_PATTERN.findall(sentence):
        for part in group.split(";"):
            part = part.strip()
            if NAME_PATTERN.search(part):
                citations.append(part)
    return citations


def parse_manuscript(text):
    sentences = split_sentences(text)
    parsed = []
    for s in sentences:
        parsed.append({
            "sentence": s,
            "citations": extract_citations(s),
        })
    return parsed
