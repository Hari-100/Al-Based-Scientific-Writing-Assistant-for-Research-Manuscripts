from citeguard.aggregator import PRIORITY

# one plain vocabulary for how well each sentence is supported
LEVELS = {
    "verified": "Supported",
    "weak_support": "Weak support",
    "possible_mismatch": "Check source",
    "unresolved": "Source not found",
    "exists_no_abstract": "Cannot check",
    "uncited_claim": "No citation",
}


def annotate(parsed, report):
    items_by_sentence = {}
    for idx, item in enumerate(report):
        items_by_sentence.setdefault(item["sentence"], []).append(idx)

    annotated = []
    for p in parsed:
        idxs = items_by_sentence.get(p["sentence"], [])
        status = None
        if idxs:
            # a sentence with several citations shows its worst status
            status = min((report[i]["status"] for i in idxs), key=lambda s: PRIORITY.get(s, 9))
        annotated.append({
            "sentence": p["sentence"],
            "status": status,
            "level": LEVELS.get(status, ""),
            "items": idxs,
        })
    return annotated