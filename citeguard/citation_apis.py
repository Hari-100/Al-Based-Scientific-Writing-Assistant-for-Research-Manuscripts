import re
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_URL = "https://api.crossref.org/works"
ARXIV_URL = "http://export.arxiv.org/api/query"

YEAR_PATTERN = re.compile(r"(\d{4})")
SURNAME_PATTERN = re.compile(r"[A-Z][a-zA-Z\-]+")


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


def _year_matches(citation_text, result_year):
    cited_year = _extract_year(citation_text)
    if cited_year is None or result_year is None:
        return True  # cannot check, do not penalise
    return abs(cited_year - result_year) <= 1


def search_semantic_scholar(query, timeout=10):
    params = {"query": query, "limit": 1, "fields": "title,abstract,year,authors"}
    try:
        r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        paper = data[0]
        return {
            "title": paper.get("title"),
            "abstract": paper.get("abstract") or "",
            "year": paper.get("year"),
            "source": "semantic_scholar",
        }
    except requests.RequestException:
        return None


def search_crossref(query, timeout=10):
    params = {"query.bibliographic": query, "rows": 1}
    try:
        r = requests.get(CROSSREF_URL, params=params, timeout=timeout)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        title = item.get("title", [""])[0]
        abstract = item.get("abstract", "") or ""
        return {
            "title": title,
            "abstract": abstract,
            "year": item.get("published", {}).get("date-parts", [[None]])[0][0],
            "source": "crossref",
        }
    except requests.RequestException:
        return None


def search_arxiv(citation_text, timeout=10):
    # arxiv's free text search is noisy, searching by author surname is much more precise
    surname = _extract_surname(citation_text)
    search_query = f"au:{surname}" if surname else f"all:{_clean_query(citation_text)}"
    params = {"search_query": search_query, "start": 0, "max_results": 3}
    try:
        r = requests.get(ARXIV_URL, params=params, timeout=timeout)
        r.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
        entries = root.findall("atom:entry", ns)
        if not entries:
            return None

        cited_year = _extract_year(citation_text)
        # among the top matches, prefer one whose year actually matches the citation
        best = None
        for entry in entries:
            published = entry.find("atom:published", ns).text[:4]
            year = int(published)
            candidate = {
                "title": entry.find("atom:title", ns).text.strip(),
                "abstract": entry.find("atom:summary", ns).text.strip(),
                "year": year,
                "source": "arxiv",
            }
            if best is None:
                best = candidate
            if cited_year is not None and abs(year - cited_year) <= 1:
                return candidate
        return best
    except (requests.RequestException, ET.ParseError):
        return None


@lru_cache(maxsize=256)
def resolve_citation(citation_text):
    query = _clean_query(citation_text)
    # try all three sources, more coverage than any single API alone
    candidates = []
    for search_fn in (search_semantic_scholar, search_crossref, search_arxiv):
        arg = citation_text if search_fn is search_arxiv else query
        result = search_fn(arg)
        if result:
            result["year_match"] = _year_matches(citation_text, result.get("year"))
            candidates.append(result)

    if not candidates:
        return None

    # prefer a candidate with both an abstract and a matching year
    for c in candidates:
        if c["abstract"] and c["year_match"]:
            return c
    for c in candidates:
        if c["abstract"]:
            return c
    return candidates[0]