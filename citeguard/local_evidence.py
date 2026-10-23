"""Load a small, offline, curated evidence corpus into CiteGuard's vector store.

The records are paraphrased research notes with source links, not verbatim abstracts.
This library is intended for uncited-claim retrieval only; it must not be used to
confirm the identity of an in-text citation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_LIBRARY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "local_evidence.json"
)


def load_local_evidence(store: Any, library_path: str | Path | None = None) -> int:
    """Add curated local notes to ``store`` and return the number loaded.

    Expected store interface: ``store.add(text, metadata)``. No API calls are made.
    Call this after citation verification so the library only enriches claim RAG.
    """
    path = Path(library_path) if library_path is not None else DEFAULT_LIBRARY_PATH
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("local evidence library 'records' must be a list")

    loaded = 0
    seen_ids: set[str] = set()
    for entry in records:
        required = ("id", "title", "citation", "year", "url", "summary", "topics")
        missing = [key for key in required if not entry.get(key)]
        if missing:
            raise ValueError(
                f"Local evidence record {entry.get('id', '<unknown>')} missing: "
                + ", ".join(missing)
            )
        record_id = str(entry["id"])
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)

        topics = entry.get("topics", [])
        # Include the title and topical terms in the indexed text to make retrieval
        # robust even when the body of a user's query uses a model's common name.
        document_text = (
            f"{entry['title']}. "
            f"Evidence summary (paraphrased): {entry['summary']} "
            f"Topics: {', '.join(str(topic) for topic in topics)}."
        )
        metadata = {
            "title": str(entry["title"]),
            "citation": str(entry["citation"]),
            "authors": entry.get("authors", []),
            "year": entry["year"],
            "source": "local_evidence_library",
            "source_url": str(entry["url"]),
            "text_kind": "curated_paraphrase",
            "topics": topics,
            "record_id": record_id,
        }
        store.add(document_text, metadata)
        loaded += 1

    return loaded
