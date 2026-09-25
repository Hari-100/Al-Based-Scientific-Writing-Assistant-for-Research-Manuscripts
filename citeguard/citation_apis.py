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
    # "et al." and commas are noise to a search engine
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
        return True

    return abs(cited_year - result_year) <= 1


def _author_matches(citation_text, authors):
    """
    Check whether the cited surname appears among the paper authors.
    """
    surname = _extract_surname(citation_text)

    if not surname or not authors:
        return False

    surname = surname.lower()

    for author in authors:
        if isinstance(author, dict):
            name = author.get("name", "")
        else:
            name = str(author)

        if surname in name.lower():
            return True

    return False


def _candidate_score(citation_text, candidate):
    """
    Rank a paper candidate.

    Author + year are the strongest signals because a citation such as
    "Vaswani et al., 2017" contains very little title information.
    """

    cited_year = _extract_year(citation_text)

    score = 0

    # Strong signal: cited author actually appears on the paper.
    if candidate.get("author_match"):
        score += 100

    # Strong signal: publication year matches.
    result_year = candidate.get("year")

    if cited_year is not None and result_year is not None:
        if result_year == cited_year:
            score += 80
        elif abs(result_year - cited_year) == 1:
            score += 30

    # Prefer papers with abstracts because CiteGuard needs them later.
    if candidate.get("abstract"):
        score += 10

    # Small preference for a non-empty title.
    if candidate.get("title"):
        score += 2

    return score


def search_semantic_scholar(query, timeout=10):
    """
    Search Semantic Scholar and return multiple candidates.

    Previously only the first result was returned, which caused unrelated
    papers to be accepted for citations such as "Vaswani et al., 2017".
    """

    params = {
        "query": query,
        "limit": 10,
        "fields": "title,abstract,year,authors",
    }

    try:
        r = requests.get(
            SEMANTIC_SCHOLAR_URL,
            params=params,
            timeout=timeout,
        )
        r.raise_for_status()

        data = r.json().get("data", [])

        candidates = []

        for paper in data:
            authors = paper.get("authors") or []

            candidates.append({
                "title": paper.get("title"),
                "abstract": paper.get("abstract") or "",
                "year": paper.get("year"),
                "authors": authors,
                "source": "semantic_scholar",
            })

        return candidates

    except (requests.RequestException, ValueError):
        return []


def search_crossref(query, timeout=10):
    """
    Search CrossRef and return multiple candidates.
    """

    params = {
        "query.bibliographic": query,
        "rows": 10,
    }

    try:
        r = requests.get(
            CROSSREF_URL,
            params=params,
            timeout=timeout,
        )
        r.raise_for_status()

        items = r.json().get("message", {}).get("items", [])

        candidates = []

        for item in items:
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""

            abstract = item.get("abstract", "") or ""

            date_parts = (
                item.get("published", {})
                .get("date-parts", [[None]])
            )

            year = date_parts[0][0] if date_parts and date_parts[0] else None

            authors = item.get("author", []) or []

            candidates.append({
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors,
                "source": "crossref",
            })

        return candidates

    except (requests.RequestException, ValueError):
        return []


def search_arxiv(citation_text, timeout=10):
    """
    Search arXiv by author surname and return several candidates.
    """

    surname = _extract_surname(citation_text)

    if surname:
        search_query = f"au:{surname}"
    else:
        search_query = f"all:{_clean_query(citation_text)}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": 10,
    }

    try:
        r = requests.get(
            ARXIV_URL,
            params=params,
            timeout=timeout,
        )
        r.raise_for_status()

        ns = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        root = ET.fromstring(r.content)
        entries = root.findall("atom:entry", ns)

        candidates = []

        for entry in entries:
            published_element = entry.find("atom:published", ns)
            title_element = entry.find("atom:title", ns)
            abstract_element = entry.find("atom:summary", ns)

            if published_element is None:
                continue

            year = int(published_element.text[:4])

            authors = []

            for author_element in entry.findall("atom:author", ns):
                name_element = author_element.find("atom:name", ns)

                if name_element is not None:
                    authors.append({
                        "name": name_element.text
                    })

            candidates.append({
                "title": (
                    title_element.text.strip()
                    if title_element is not None
                    else ""
                ),
                "abstract": (
                    abstract_element.text.strip()
                    if abstract_element is not None
                    else ""
                ),
                "year": year,
                "authors": authors,
                "source": "arxiv",
            })

        return candidates

    except (requests.RequestException, ET.ParseError, ValueError):
        return []


@lru_cache(maxsize=256)
def resolve_citation(citation_text):
    """
    Resolve an author-year citation across Semantic Scholar, CrossRef,
    and arXiv.

    The resolver now evaluates multiple candidates instead of blindly
    accepting the first API result.
    """

    query = _clean_query(citation_text)

    all_candidates = []

    # Semantic Scholar
    semantic_candidates = search_semantic_scholar(query)

    # CrossRef
    crossref_candidates = search_crossref(query)

    # arXiv
    arxiv_candidates = search_arxiv(citation_text)

    all_candidates.extend(semantic_candidates)
    all_candidates.extend(crossref_candidates)
    all_candidates.extend(arxiv_candidates)

    if not all_candidates:
        return None

    # Add matching metadata to every candidate.
    for candidate in all_candidates:
        candidate["year_match"] = _year_matches(
            citation_text,
            candidate.get("year"),
        )

        candidate["author_match"] = _author_matches(
            citation_text,
            candidate.get("authors"),
        )

        candidate["_score"] = _candidate_score(
            citation_text,
            candidate,
        )

    # ---------------------------------------------------------
    # First preference:
    # exact author + exact year
    # ---------------------------------------------------------

    cited_year = _extract_year(citation_text)

    exact_candidates = [
        c for c in all_candidates
        if c.get("author_match")
        and cited_year is not None
        and c.get("year") == cited_year
    ]

    if exact_candidates:
        best = max(
            exact_candidates,
            key=lambda c: c["_score"]
        )

        best.pop("_score", None)
        best.pop("authors", None)
        best.pop("author_match", None)

        return best

    # ---------------------------------------------------------
    # Second preference:
    # author + acceptable year (±1)
    # ---------------------------------------------------------

    author_year_candidates = [
        c for c in all_candidates
        if c.get("author_match")
        and c.get("year_match")
    ]

    if author_year_candidates:
        best = max(
            author_year_candidates,
            key=lambda c: c["_score"]
        )

        best.pop("_score", None)
        best.pop("authors", None)
        best.pop("author_match", None)

        return best

    # ---------------------------------------------------------
    # Third preference:
    # any matching-year candidate
    # ---------------------------------------------------------

    year_candidates = [
        c for c in all_candidates
        if c.get("year_match")
    ]

    if year_candidates:
        best = max(
            year_candidates,
            key=lambda c: c["_score"]
        )

        best.pop("_score", None)
        best.pop("authors", None)
        best.pop("author_match", None)

        return best

    # ---------------------------------------------------------
    # Final fallback:
    # return the strongest candidate even though its year may
    # not match. citation_verifier.py will mark it as
    # possible_mismatch.
    # ---------------------------------------------------------

    best = max(
        all_candidates,
        key=lambda c: c["_score"]
    )

    best.pop("_score", None)
    best.pop("authors", None)
    best.pop("author_match", None)

    return best