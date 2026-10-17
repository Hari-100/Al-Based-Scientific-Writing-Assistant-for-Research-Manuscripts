"""Print parser output and resolved source diagnostics without running the LLM."""
import sys
from citeguard.parser import parse_manuscript
from citeguard.citation_apis import resolve_citation


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "demo_manuscript.txt"
    with open(path, encoding="utf-8") as file:
        text = file.read()

    parsed = parse_manuscript(text)
    citation_count = sum(len(item["citations"]) for item in parsed)
    print(f"Sentences: {len(parsed)} | Explicit author-year citations: {citation_count}\n")

    for item in parsed:
        for citation in item["citations"]:
            result = resolve_citation(citation, item["sentence"])
            print(f"CITATION: {citation}")
            print(f"SENTENCE: {item['sentence']}")
            if result is None:
                print("RESULT: NO RELIABLE MATCH")
            else:
                print(f"TITLE: {result.get('title')}")
                print(f"YEAR: {result.get('year')}")
                print(f"SOURCE: {result.get('source')} | ALL SOURCES: {result.get('sources')}")
                print(f"AUTHOR MATCH: {result.get('author_match')} | YEAR MATCH: {result.get('year_match')}")
                print(f"CONTEXT SCORE: {result.get('context_score')} | CONTEXT HITS: {result.get('context_hits')}")
                print(f"ABSTRACT CHARACTERS: {len(result.get('abstract') or '')}")
            print("-" * 78)


if __name__ == "__main__":
    main()
