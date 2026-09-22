# priority order, lower number surfaces first in the report
PRIORITY = {
    "unresolved": 0,
    "possible_mismatch": 1,
    "uncited_claim": 2,
    "weak_support": 3,
    "exists_no_abstract": 4,
    "verified": 5,
}


def aggregate(citation_results, claim_results):
    items = citation_results + claim_results
    items.sort(key=lambda x: PRIORITY.get(x["status"], 9))
    return items


def to_text_report(items):
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. [{item['status'].upper()}] {item['sentence']}")
        if item["status"] in ("unresolved", "exists_no_abstract", "possible_mismatch"):
            lines.append(f"   note: {item.get('note', '')}")
        elif item["status"] in ("verified", "weak_support"):
            if "reason" in item:
                lines.append(f"   matched: {item.get('title', '')}, llm reason: {item['reason']}")
            else:
                lines.append(f"   matched: {item.get('title', '')} (similarity {item.get('similarity', '?')})")
        elif item["status"] == "uncited_claim":
            if item.get("llm_note"):
                lines.append(f"   llm suggestion: {item['llm_note']}")
            elif item["suggestions"]:
                titles = ", ".join(s["title"] for s in item["suggestions"])
                lines.append(f"   candidate sources: {titles}")
            else:
                lines.append("   candidate sources: none found yet")
        lines.append("")
    return "\n".join(lines)
