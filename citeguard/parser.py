"""Small author-year manuscript parser for CiteGuard.

Supported citation format: (Vaswani et al., 2017), including multiple
standard author-year citations in the same parentheses. Numeric references,
reference-list citation resolution, and unusual year suffix shorthand are
intentionally out of scope for this mini-project.
"""

import re

REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(?:(?:\d+(?:\.\d+)*)\.?\s+)?"
    r"(?:references|bibliography|works cited|literature cited)\s*:?[ \t]*$"
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# One author plus optional "et al.", "and Other", or "& Other".
# The year suffix (a/b/c) is deliberately ignored: it is only needed to
# disambiguate entries with a bibliography, which this MVP does not require.
AUTHOR_YEAR = re.compile(
    r"(?P<authors>"
    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|"
    r"&\s*[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+))?"
    r")\s*,\s*(?P<year>(?:19|20)\d{2})[a-z]?\b"
)


def _normalize_text(text):
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Join words hyphenated by PDF line wrapping: "trans-\nformer".
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return text.strip()


def _split_references(text):
    match = REFERENCE_HEADING.search(text)
    if not match:
        return text, ""
    return text[:match.start()].strip(), text[match.start():].strip()


def _split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Lightweight splitter, no downloaded NLP package required.
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ“‘\"'])", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _extract_citation_records(sentence):
    """Return one citation record per explicit author-year citation.

    The local context is important when a sentence lists several models, e.g.
    "GPT (Radford et al., 2019), BERT (Devlin et al., 2019)". Passing the
    whole long sentence to both searches makes the two works harder to tell apart.
    """
    records = []
    seen = set()
    for group in re.finditer(r"\(([^()]*)\)", sentence):
        content = group.group(1)
        left = sentence[max(0, group.start() - 120):group.start()]
        right = sentence[group.end():min(len(sentence), group.end() + 35)]
        local_context = re.sub(r"\([^()]*\)", " ", left + " " + right)
        local_context = re.sub(r"\s+", " ", local_context).strip()

        for match in AUTHOR_YEAR.finditer(content):
            authors = re.sub(r"\s+", " ", match.group("authors")).strip()
            year = match.group("year")
            citation = f"{authors}, {year}"
            key = citation.casefold()
            if key not in seen:
                seen.add(key)
                records.append({"citation": citation, "context": local_context})
    return records


def _extract_citations(sentence, reference_map=None):
    """Compatibility helper returning citation strings."""
    return [record["citation"] for record in _extract_citation_records(sentence)]

def parse_manuscript(text):
    """Return [{"sentence": str, "citations": [str, ...]}, ...]."""
    normalized = _normalize_text(text)
    body, _references = _split_references(normalized)
    # A single line break usually comes from PDF wrapping; blank lines remain
    # paragraph boundaries. Citation extraction itself is sentence based.
    body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)

    parsed = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _split_sentences(paragraph):
            citation_records = _extract_citation_records(sentence)
            citations = [record["citation"] for record in citation_records]
            parsed.append({
                "sentence": sentence,
                "citations": citations,
                "citation_contexts": citation_records,
            })
    return parsed
