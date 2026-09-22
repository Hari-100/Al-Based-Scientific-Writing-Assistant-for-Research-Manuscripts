import numpy as np
from citeguard.citation_apis import resolve_citation
from citeguard.embeddings import embed
from citeguard.llm import llm_available, judge_support

SUPPORT_THRESHOLD = 0.35  # used only when the llm is unavailable


def verify_citation(sentence, citation_text, store):
    paper = resolve_citation(citation_text)

    if paper is None:
        return {
            "citation": citation_text,
            "status": "unresolved",
            "note": "no matching paper found in Semantic Scholar, CrossRef, or arXiv, possible fabricated citation",
        }

    if not paper["abstract"]:
        return {
            "citation": citation_text,
            "status": "exists_no_abstract",
            "note": "paper found but no abstract available to check support",
            "title": paper["title"],
        }

    if not paper.get("year_match", True):
        store.add(paper["abstract"], {"title": paper["title"], "citation": citation_text})
        return {
            "citation": citation_text,
            "status": "possible_mismatch",
            "note": f"found '{paper['title']}' but its year does not match the citation, verify this is the right paper",
            "title": paper["title"],
        }

    # cache the retrieved paper so claim verification can search it later
    store.add(paper["abstract"], {"title": paper["title"], "citation": citation_text})

    # llm judges support first, embedding similarity is the fallback if it is unavailable
    if llm_available():
        verdict = judge_support(sentence, paper["abstract"])
        if verdict["supported"] is not None:
            status = "verified" if verdict["supported"] else "weak_support"
            return {
                "citation": citation_text,
                "status": status,
                "reason": verdict["reason"],
                "title": paper["title"],
                "method": "llm",
            }

    sent_vec = embed([sentence])[0]
    abs_vec = embed([paper["abstract"]])[0]
    similarity = float(np.dot(sent_vec, abs_vec))
    status = "verified" if similarity >= SUPPORT_THRESHOLD else "weak_support"
    return {
        "citation": citation_text,
        "status": status,
        "similarity": round(similarity, 3),
        "title": paper["title"],
        "method": "embedding",
    }


def verify_all(parsed_sentences, store):
    results = []
    for item in parsed_sentences:
        for citation in item["citations"]:
            result = verify_citation(item["sentence"], citation, store)
            result["sentence"] = item["sentence"]
            results.append(result)
    return results
