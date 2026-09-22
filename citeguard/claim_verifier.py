import re
from citeguard.llm import llm_available, suggest_for_claim

# basic markers of a checkable factual or empirical claim
NUMBER_PATTERN = re.compile(r"\d+(\.\d+)?\s*%?")
CLAIM_WORDS = [
    "significantly", "outperforms", "improves", "reduces", "increases",
    "decreases", "always", "never", "proves", "demonstrates", "shows that",
]


def looks_like_claim(sentence):
    if NUMBER_PATTERN.search(sentence):
        return True
    lowered = sentence.lower()
    return any(word in lowered for word in CLAIM_WORDS)


def check_claims(parsed_sentences, store, k=2):
    flagged = []
    for item in parsed_sentences:
        if item["citations"]:
            continue  # already has a citation, citation_verifier handles it
        if not looks_like_claim(item["sentence"]):
            continue
        suggestions = store.search(item["sentence"], k=k)
        llm_note = None
        if suggestions and llm_available():
            llm_note = suggest_for_claim(item["sentence"], suggestions)
        flagged.append({
            "sentence": item["sentence"],
            "status": "uncited_claim",
            "suggestions": suggestions,
            "llm_note": llm_note,
        })
    return flagged
