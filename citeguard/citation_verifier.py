"""Verify citations after conservative metadata resolution."""

import numpy as np

from citeguard.citation_apis import resolve_citation
from citeguard.embeddings import embed
from citeguard.llm import llm_available, judge_support

SUPPORT_THRESHOLD = 0.35


def verify_citation(sentence, citation_text, store, citation_context=None):
    # Resolve from the nearby model/claim context, but judge support against the
    # complete sentence so the verdict is based on the full authored claim.
    paper = resolve_citation(citation_text, citation_context or sentence)

    if paper is None:
        return {
            "citation": citation_text,
            "status": "unresolved",
            "note": "No sufficiently reliable scholarly match was found. Check the author-year citation or provide a full paper title.",
        }

    base = {
        "citation": citation_text,
        "title": paper.get("title", "Unknown title"),
        "year": paper.get("year"),
        "source": paper.get("source"),
        "sources": paper.get("sources", []),
        "doi": paper.get("doi"),
        "url": paper.get("url"),
        "context_score": paper.get("context_score"),
    }

    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return {
            **base,
            "status": "exists_no_abstract",
            "note": "A contextually plausible paper record was found, but metadata providers did not supply an abstract. Citation identity is not the same as claim verification.",
        }

    store.add(abstract, {"title": paper.get("title", ""), "citation": citation_text})

    if llm_available():
        verdict = judge_support(sentence, abstract)
        if verdict and verdict.get("supported") is not None:
            return {
                **base,
                "status": "verified" if verdict["supported"] else "weak_support",
                "reason": verdict.get("reason", ""),
                "method": "llm",
            }

    # Graceful local fallback when the LLM is not configured or returns no verdict.
    sent_vec = embed([sentence])[0]
    abs_vec = embed([abstract])[0]
    similarity = float(np.dot(sent_vec, abs_vec))
    return {
        **base,
        "status": "verified" if similarity >= SUPPORT_THRESHOLD else "weak_support",
        "similarity": round(similarity, 3),
        "method": "embedding",
        "note": "Embedding similarity is a fallback signal, not proof of factual support.",
    }


def verify_all(parsed_sentences, store):
    results = []
    for item in parsed_sentences:
        # Use citation-local context to disambiguate lists such as GPT (Radford),
        # BERT (Devlin), and RoBERTa (Liu) within the same sentence.
        context_by_citation = {
            record.get("citation", "").casefold(): record.get("context", "")
            for record in item.get("citation_contexts", [])
        }
        for citation in item.get("citations", []):
            result = verify_citation(
                item["sentence"],
                citation,
                store,
                citation_context=context_by_citation.get(citation.casefold()),
            )
            result["sentence"] = item["sentence"]
            results.append(result)
    return results
