"""Conservative, cached scholarly-metadata lookup for author-year citations.

Provider order:
  1. OpenAlex (broad coverage; abstract reconstructed from inverted index)
  2. OpenAIRE Graph (broad aggregated research metadata)
  3. Crossref (useful bibliographic fallback; abstracts are not guaranteed)

A candidate is NOT accepted just because its author/year metadata happens to
match. It also needs contextual evidence from the title/abstract. This avoids
false matches such as an unrelated 2017 Crossref record being selected for
(Vaswani et al., 2017).
"""

import html
import os
import re
import unicodedata
from functools import lru_cache

import requests

OPENALEX_URL = "https://api.openalex.org/works"
OPENAIRE_URL = "https://api.openaire.eu/graph/v4/research-products"
CROSSREF_URL = "https://api.crossref.org/works"
TIMEOUT = 5

YEAR_RE = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b", re.I)
AUTHOR_RE = re.compile(r"^\s*([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "of", "in", "on", "to", "as", "at", "have", "has", "had", "are", "was", "were",
    "been", "being", "this", "that", "these", "those", "with", "from",
    "into", "than", "then", "them", "they", "their", "there", "which",
    "while", "when", "where", "what", "who", "whose", "such", "using",
    "used", "use", "can", "could", "would", "should", "may", "might",
    "also", "more", "most", "many", "much", "some", "each", "both",
    "all", "any", "not", "but", "between", "through", "across", "over",
    "under", "from", "into", "its", "our", "their", "than", "since",
    "show", "shows", "showed", "based", "approach", "method", "methods",
    "model", "models", "paper", "study", "work", "works", "task", "tasks",
    "result", "results", "proposed", "present", "using", "large", "scale",
}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "CiteGuard/0.3 (student research verification demo)"})


def _fold(text):
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower()


def _year(citation_text):
    match = YEAR_RE.search(citation_text or "")
    return int(match.group(1)) if match else None


def _surname(citation_text):
    match = AUTHOR_RE.search(citation_text or "")
    return _fold(match.group(1)) if match else None


def _tokens(text):
    tokens = []
    for token in TOKEN_RE.findall(_fold(text)):
        token = token.lower()
        # Light plural normalization helps "Transformer" match "Transformers"
        # without introducing a heavyweight NLP dependency.
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        if len(token) >= 2 and token not in STOPWORDS:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def _context(sentence):
    # Remove parenthetical citations, not meaningful model names such as BERT.
    stripped = re.sub(r"\([^()]*\)", " ", sentence or "")
    return _tokens(stripped)


def _rebuild_openalex_abstract(index):
    if not isinstance(index, dict) or not index:
        return ""
    words = {}
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words[position] = word
    if not words:
        return ""
    return " ".join(words[i] for i in range(max(words) + 1) if i in words)


def _clean_abstract(value):
    if isinstance(value, dict):
        # Some aggregators return language-tagged description objects.
        value = value.get("description") or value.get("value") or value.get("text") or value.get("content") or ""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _normalize_candidate(title="", abstract="", year=None, authors=None, source="unknown", doi=None, url=None):
    try:
        year = int(str(year)[:4]) if year else None
    except (TypeError, ValueError):
        year = None
    return {
        "title": str(title or "").strip(),
        "abstract": _clean_abstract(abstract),
        "year": year,
        "authors": [str(a).strip() for a in (authors or []) if str(a or "").strip()],
        "source": source,
        "doi": doi,
        "url": url,
        "sources": [source],
    }


def _openalex_headers_params():
    params = {
        "per-page": 10,
        "select": "id,doi,display_name,publication_year,authorships,abstract_inverted_index,ids,primary_location",
    }
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    mailto = os.getenv("OPENALEX_EMAIL", "").strip()
    if mailto:
        params["mailto"] = mailto
    return params


@lru_cache(maxsize=256)
def _search_openalex(query):
    params = _openalex_headers_params()
    params["search"] = query
    try:
        response = _SESSION.get(OPENALEX_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        records = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return []

    candidates = []
    for record in records:
        authors = []
        for authorship in record.get("authorships") or []:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)
        ids = record.get("ids") or {}
        doi = record.get("doi") or ids.get("doi")
        primary = record.get("primary_location") or {}
        landing = primary.get("landing_page_url")
        candidates.append(_normalize_candidate(
            title=record.get("display_name"),
            abstract=_rebuild_openalex_abstract(record.get("abstract_inverted_index")),
            year=record.get("publication_year"),
            authors=authors,
            source="openalex",
            doi=doi,
            url=landing or doi or record.get("id"),
        ))
    return candidates


def _openaire_year(record):
    date = record.get("publicationDate") or record.get("publication_date") or record.get("date")
    match = YEAR_RE.search(str(date or ""))
    return int(match.group(1)) if match else None


def _openaire_authors(record):
    authors = record.get("authors") or record.get("author") or []
    out = []
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, dict):
                name = author.get("fullName") or author.get("fullname") or author.get("name")
                if not name:
                    name = " ".join(x for x in [author.get("name"), author.get("surname")] if x)
            else:
                name = str(author or "")
            if name:
                out.append(name)
    return out


def _openaire_abstract(record):
    descriptions = record.get("descriptions") or record.get("description") or record.get("abstract") or []
    if isinstance(descriptions, str):
        return _clean_abstract(descriptions)
    if isinstance(descriptions, dict):
        return _clean_abstract(descriptions)
    if isinstance(descriptions, list):
        # Prefer a substantial description, which is more likely to be the abstract.
        values = [_clean_abstract(item) for item in descriptions]
        values = [value for value in values if value]
        return max(values, key=len) if values else ""
    return ""


@lru_cache(maxsize=256)
def _search_openaire(query):
    params = {"search": query, "page_size": 10}
    mailto = os.getenv("OPENALEX_EMAIL", "").strip()
    if mailto:
        params["mailto"] = mailto
    try:
        response = _SESSION.get(OPENAIRE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        records = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return []

    candidates = []
    for record in records:
        pids = record.get("pids") or []
        doi = None
        for pid in pids:
            if isinstance(pid, dict) and str(pid.get("scheme") or pid.get("type") or "").lower() == "doi":
                doi = pid.get("value") or pid.get("content")
                break
        candidates.append(_normalize_candidate(
            title=record.get("mainTitle") or record.get("title"),
            abstract=_openaire_abstract(record),
            year=_openaire_year(record),
            authors=_openaire_authors(record),
            source="openaire",
            doi=doi,
            url=doi or record.get("id"),
        ))
    return candidates


@lru_cache(maxsize=256)
def _search_crossref(query, surname):
    params = {"query.bibliographic": query, "rows": 10}
    # Field queries are AND-ed by Crossref; author narrows the search instead of
    # relying on a loose substring check after an unrestricted search.
    if surname:
        params["query.author"] = surname
    mailto = os.getenv("OPENALEX_EMAIL", "").strip()
    if mailto:
        params["mailto"] = mailto
    try:
        response = _SESSION.get(CROSSREF_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        records = response.json().get("message", {}).get("items", [])
    except (requests.RequestException, ValueError, TypeError):
        return []

    candidates = []
    for record in records:
        title_values = record.get("title") or []
        title = title_values[0] if title_values else ""
        abstract = _clean_abstract(record.get("abstract") or "")
        date_parts = (
            record.get("published", {}).get("date-parts")
            or record.get("published-print", {}).get("date-parts")
            or record.get("published-online", {}).get("date-parts")
            or [[None]]
        )
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        authors = []
        for author in record.get("author") or []:
            if isinstance(author, dict):
                name = author.get("family") or ""
                if not name:
                    name = " ".join(x for x in [author.get("given"), author.get("family")] if x)
                authors.append(name)
        candidates.append(_normalize_candidate(
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            source="crossref",
            doi=record.get("DOI"),
            url=record.get("URL") or (f"https://doi.org/{record['DOI']}" if record.get("DOI") else None),
        ))
    return candidates


def _author_match(citation_text, authors):
    surname = _surname(citation_text)
    if not surname:
        return False
    for author in authors or []:
        parts = TOKEN_RE.findall(_fold(author))
        if surname in [part.lower() for part in parts]:
            return True
    return False


def _candidate_signals(candidate, citation_text, context_tokens):
    cited_year = _year(citation_text)
    candidate_year = candidate.get("year")
    author_match = _author_match(citation_text, candidate.get("authors"))
    year_delta = abs(candidate_year - cited_year) if candidate_year and cited_year else None
    year_match = year_delta is not None and year_delta <= 1

    title_tokens = set(_tokens(candidate.get("title", "")))
    abstract_tokens = set(_tokens(candidate.get("abstract", "")))
    context_set = set(context_tokens)
    title_hits = len(context_set & title_tokens)
    abstract_hits = len(context_set & abstract_tokens)
    all_hits = len(context_set & (title_tokens | abstract_tokens))

    # Favor actual sentence-paper evidence over metadata coincidences.
    score = 100 * int(author_match)
    score += 30 if year_delta == 0 else 12 if year_delta == 1 else -25 if year_delta is not None else -10
    score += 14 * title_hits + 4 * abstract_hits
    score += 2 if candidate.get("abstract") else 0

    candidate.update({
        "author_match": author_match,
        "year_match": year_match,
        "context_score": round(all_hits / max(1, len(context_set)), 3),
        "context_hits": all_hits,
        "title_hits": title_hits,
        "abstract_hits": abstract_hits,
        "_score": score,
    })
    return candidate


def _reliable(candidate):
    """Reject title/year coincidences without author or strong title evidence."""
    if not candidate.get("title") or not candidate.get("year_match"):
        return False
    author_match = candidate.get("author_match", False)
    title_hits = candidate.get("title_hits", 0)
    abstract_hits = candidate.get("abstract_hits", 0)
    context_hits = candidate.get("context_hits", 0)

    if author_match:
        # For records with abstracts, require some evidence the actual paper is
        # about the cited sentence. Without an abstract, only accept contextual
        # title evidence, rather than author/year coincidence alone.
        if candidate.get("abstract"):
            return context_hits >= 2
        return title_hits >= 1

    # Some aggregators omit author lists. In that case require unusually strong
    # title/context agreement; don't treat missing authors as a positive match.
    return title_hits >= 2 and abstract_hits >= 2 and context_hits >= 4


def _dedupe_candidates(candidates):
    merged = {}
    for candidate in candidates:
        title = re.sub(r"\W+", " ", _fold(candidate.get("title", ""))).strip()
        if not title:
            continue
        key = (title, candidate.get("year"))
        if key not in merged:
            merged[key] = candidate.copy()
            continue
        existing = merged[key]
        if len(candidate.get("abstract", "")) > len(existing.get("abstract", "")):
            existing["abstract"] = candidate["abstract"]
        existing["authors"] = list(dict.fromkeys(existing.get("authors", []) + candidate.get("authors", [])))
        existing["sources"] = list(dict.fromkeys(existing.get("sources", [existing.get("source")]) + candidate.get("sources", [candidate.get("source")])))
        if not existing.get("doi") and candidate.get("doi"):
            existing["doi"] = candidate["doi"]
        if not existing.get("url") and candidate.get("url"):
            existing["url"] = candidate["url"]
    return list(merged.values())


def _query_for(citation_text, sentence):
    context_tokens = _context(sentence)
    surname = _surname(citation_text) or ""
    year = str(_year(citation_text) or "")
    # Keep the query reasonably short and emphasize terms from this sentence.
    query = " ".join([surname, year] + context_tokens[:12]).strip()
    return query, context_tokens


@lru_cache(maxsize=512)
def resolve_citation(citation_text, sentence=""):
    """Return a high-confidence scholarly record, or None when evidence is weak."""
    query, context_tokens = _query_for(citation_text, sentence)
    surname = _surname(citation_text)

    # Cascade providers. Stop early if OpenAlex provides a reliable abstract;
    # otherwise try the other aggregate registries. Cache makes reruns much faster.
    candidates = _search_openalex(query)
    ranked = [_candidate_signals(c.copy(), citation_text, context_tokens) for c in candidates]
    ranked = _dedupe_candidates(ranked)
    reliable = [c for c in ranked if _reliable(c)]
    reliable_with_abstract = [c for c in reliable if c.get("abstract")]
    if reliable_with_abstract:
        best = max(reliable_with_abstract, key=lambda c: c["_score"])
        return _public_result(best)

    candidates += _search_openaire(query)
    ranked = [_candidate_signals(c.copy(), citation_text, context_tokens) for c in _dedupe_candidates(candidates)]
    reliable = [c for c in ranked if _reliable(c)]
    reliable_with_abstract = [c for c in reliable if c.get("abstract")]
    if reliable_with_abstract:
        best = max(reliable_with_abstract, key=lambda c: c["_score"])
        return _public_result(best)

    candidates += _search_crossref(query, surname)
    ranked = [_candidate_signals(c.copy(), citation_text, context_tokens) for c in _dedupe_candidates(candidates)]
    reliable = [c for c in ranked if _reliable(c)]
    if reliable:
        best = max(reliable, key=lambda c: (bool(c.get("abstract")), c["_score"]))
        return _public_result(best)

    return None


def _public_result(candidate):
    return {
        "title": candidate.get("title", ""),
        "abstract": candidate.get("abstract", ""),
        "year": candidate.get("year"),
        "source": candidate.get("source", "unknown"),
        "sources": candidate.get("sources", [candidate.get("source", "unknown")]),
        "doi": candidate.get("doi"),
        "url": candidate.get("url"),
        "author_match": candidate.get("author_match", False),
        "year_match": candidate.get("year_match", False),
        "context_score": candidate.get("context_score", 0.0),
        "context_hits": candidate.get("context_hits", 0),
        "title_hits": candidate.get("title_hits", 0),
    }
