import re

from citeguard.llm import llm_available, suggest_for_claim

NUMBER_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|percentage)?\b", re.I)
NUMERIC_EVIDENCE_WORDS = {
    "accuracy", "precision", "recall", "f1", "auc", "bleu", "rouge", "score",
    "error", "errors", "rate", "latency", "runtime", "improvement", "reduction",
    "increase", "decrease", "outperforms", "baseline", "benchmark", "documents",
    "samples", "examples", "instances", "queries", "images", "records", "papers",
    "tokens", "parameters", "epochs", "datasets", "dataset", "percent", "percentage",
    "million", "billion", "thousand", "times", "fold", "hours", "minutes", "seconds",
    "memory", "throughput", "speed", "performance", "loss", "size", "capacity",
}
CLAIM_PHRASES = (
    "significantly", "outperforms", "improves", "reduces", "increases", "decreases",
    "always", "never", "proves", "demonstrates", "shows that", "achieves", "achieved",
    "results in", "leads to", "is superior", "is more accurate", "is faster", "is better",
)




def _deduplicate_suggestions(candidates):
    """Keep the highest-ranked suggestion for each normalized paper title."""
    unique = []
    seen = set()
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        title = str(candidate.get("title") or "").strip().casefold()
        key = re.sub(r"[^a-z0-9]+", "", title)
        if not key:
            body = str(candidate.get("text") or "").strip().casefold()
            key = "text:" + re.sub(r"[^a-z0-9]+", "", body[:180])
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def looks_like_claim(sentence):
    text = str(sentence or "").strip()
    lowered = text.lower()
    if not text:
        return False

    # Do not treat figure/table/section labels or isolated year mentions as evidence claims.
    if re.match(r"^(?:figure|fig\.?|table|section|chapter|equation)\s+\d+\b", lowered):
        return False

    if any(phrase in lowered for phrase in CLAIM_PHRASES):
        return True

    if NUMBER_PATTERN.search(lowered):
        words = set(re.findall(r"[a-z]+", lowered))
        # Numbers alone are too noisy: require a measurement or experimental-result cue.
        if words & NUMERIC_EVIDENCE_WORDS:
            return True
        if re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent|percentage)\b", lowered):
            return True

    return False


def check_claims(parsed_sentences, store, k=3):
    flagged = []
    for item in parsed_sentences:
        # The parser maps author-year and numeric citations before this stage.
        if item.get("citations"):
            continue
        sentence = item.get("sentence", "")
        if not looks_like_claim(sentence):
            continue

        suggestions = store.search(sentence, k=max(k * 4, 12))
        suggestions = _deduplicate_suggestions(suggestions)[:k]
        if not suggestions:
            llm_note = "No candidate papers were retrieved, so no source suggestion is available."
        elif not llm_available():
            llm_note = "LLM suggestion unavailable: GROQ_API_KEY is not configured in this process."
        else:
            llm_note = suggest_for_claim(sentence, suggestions)
            if not llm_note:
                llm_note = "No suggestion available: the LLM returned an empty response."

        flagged.append({
            "sentence": sentence,
            "status": "uncited_claim",
            "suggestions": suggestions,
            "llm_note": llm_note,
        })
    return flagged
