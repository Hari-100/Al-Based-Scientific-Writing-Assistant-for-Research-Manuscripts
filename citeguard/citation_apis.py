import re
import unicodedata
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_URL = "https://api.crossref.org/works"
ARXIV_URL = "http://export.arxiv.org/api/query"

YEAR_PATTERN = re.compile(r"(\d{4})")
SURNAME_PATTERN = re.compile(r"[A-Z][a-zA-Z\-]+")
WORD_PATTERN = re.compile(r"[A-Za-z]{3,}")
STOPWORDS = {
    "the", "and", "for", "have", "has", "become", "since", "their", "are", "with",
    "that", "this", "these", "those", "its", "been", "was", "were", "which", "who",
    "based", "using", "used", "from", "into", "than", "can", "also",
}


def _clean_query(citation_text):
    # "et al." and commas are noise to a search engine, strip them
    cleaned = citation_text.replace("et al.", "").replace(",", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_year(citation_text):
    m = YEAR_PATTERN.search(citation_text)
    return int(m.group(1)) if m else None


def _extract_surname(citation_text):
    m = SURNAME_PATTERN.search(citation_text)
    return m.group(0) if m else None


def _fold(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _year_matches(citation_text, result_year):
    cited_year = _extract_year(citation_text)
    if cited_year is None or result_year is None:
        return True  # cannot check, do not penalise
    return abs(cited_year - result_year) <= 1


def _surname_matches(citation_text, authors):
    surname = _extract_surname(citation_text)
    if surname is None or not authors:
        return True  # cannot check, do not penalise
    return any(_fold(surname) in _fold(a) for a in authors)


def _context_words(sentence):
    # content words of the citing sentence, minus the citation itself
    stripped = re.sub(r"\([^)]*\)", "", sentence)
    words = [w.lower() for w in WORD_PATTERN.findall(stripped)]
    return list(dict.fromkeys(w for w in words if w not in STOPWORDS))


def _overlap(paper, words):
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return sum(1 for w in words if w in text)


def search_semantic_scholar(query, timeout=10):
    params = {"query": query, "limit": 5, "fields": "title,abstract,year,authors"}
    try:
        r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=timeout)
        r.raise_for_status()
        return [{
            "title": p.get("title") or "",
            "abstract": p.get("abstract") or "",
            "year": p.get("year"),
            "authors": [a.get("name") or "" for a in p.get("authors") or []],
            "source": "semantic_scholar",
        } for p in r.json().get("data") or []]
    except (requests.RequestException, ValueError):
        return []


def search_crossref(query, timeout=10):
    params = {"query.bibliographic": query, "rows": 5}
    try:
        r = requests.get(CROSSREF_URL, params=params, timeout=timeout)
        r.raise_for_status()
        papers = []
        for item in r.json().get("message", {}).get("items", []):
            parts = item.get("published", {}).get("date-parts") or [[None]]
            papers.append({
                "title": (item.get("title") or [""])[0],
                "abstract": re.sub(r"<[^>]+>", "", item.get("abstract") or "").strip(),
                "year": parts[0][0] if parts[0] else None,
                "authors": [a.get("family") or "" for a in item.get("author", [])],
                "source": "crossref",
            })
        return papers
    except (requests.RequestException, ValueError):
        return []


def _arxiv_fetch(search_query, timeout=10):
    params = {"search_query": search_query, "start": 0, "max_results": 10}
    try:
        r = requests.get(ARXIV_URL, params=params, timeout=timeout)
        r.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in ET.fromstring(r.content).findall("atom:entry", ns):
            papers.append({
                "title": " ".join(entry.find("atom:title", ns).text.split()),
                "abstract": " ".join(entry.find("atom:summary", ns).text.split()),
                "year": int(entry.find("atom:published", ns).text[:4]),
                "authors": [a.find("atom:name", ns).text or "" for a in entry.findall("atom:author", ns)],
                "source": "arxiv",
            })
        return papers
    except (requests.RequestException, ET.ParseError, AttributeError, ValueError):
        return []


def search_arxiv(citation_text, context_words=()):
    # author field search is precise, context words narrow it when one author has many papers
    surname = _extract_surname(citation_text)
    if not surname:
        return []
    if context_words:
        terms = " AND ".join(f"all:{w}" for w in context_words)
        papers = _arxiv_fetch(f"au:{surname} AND {terms}")
        if papers:
            return papers
    return _arxiv_fetch(f"au:{surname}")


@lru_cache(maxsize=256)
def resolve_citation(citation_text, sentence=""):
    context = _context_words(sentence)
    query = " ".join([_clean_query(citation_text)] + context[:6])

    # pool candidates from every source, then rank them in one place
    candidates = (
        search_semantic_scholar(query)
        + search_crossref(query)
        + search_arxiv(citation_text, context[:2])
    )
    if not candidates:
        return None

    for c in candidates:
        c["year_match"] = _year_matches(citation_text, c["year"])
        c["surname_match"] = _surname_matches(citation_text, c["authors"])

    # author and year must fit, then similarity to the citing sentence breaks author/year collisions
    return max(
        candidates,
        key=lambda c: (c["surname_match"], c["year_match"], _overlap(c, context), bool(c["abstract"])),
    )