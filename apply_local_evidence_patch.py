"""Safely integrate CiteGuard's local evidence library.

Run from the project root after extracting this ZIP into that root:
    python apply_local_evidence_patch.py

The script validates planned edits before writing files, creates backups, and is
intended to tolerate small differences in claim_verifier.py formatting.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "citeguard" / "pipeline.py"
CLAIM_VERIFIER = ROOT / "citeguard" / "claim_verifier.py"
LOCAL_MODULE = ROOT / "citeguard" / "local_evidence.py"
LIBRARY = ROOT / "data" / "local_evidence.json"


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_local_evidence.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def patch_pipeline(text: str) -> str:
    if "from citeguard.local_evidence import load_local_evidence" not in text:
        import_markers = (
            "from citeguard.claim_verifier import check_claims",
            "from citeguard.aggregator import aggregate",
        )
        inserted = False
        for marker in import_markers:
            if marker in text:
                text = text.replace(
                    marker,
                    marker + "\nfrom citeguard.local_evidence import load_local_evidence",
                    1,
                )
                inserted = True
                break
        if not inserted:
            raise ValueError(
                "Could not find a safe import insertion point in pipeline.py. "
                "No project files were changed."
            )

    if "load_local_evidence(store)" not in text:
        # Most reliable insertion point: after citation verification, immediately
        # before claim suggestions are generated.
        claim_call = re.compile(
            r"(?m)^(?P<indent>[ \t]*)[A-Za-z_]\w*\s*=\s*check_claims\s*\("
        )
        match = claim_call.search(text)
        if match:
            indent = match.group("indent")
            addition = (
                indent + "# Offline evidence notes enrich uncited-claim RAG only.\n"
                + indent + "load_local_evidence(store)\n"
            )
            text = text[:match.start()] + addition + text[match.start():]
        else:
            # Fallback to inserting after the known citation verification statement.
            verify_call = re.compile(
                r"(?m)^(?P<indent>[ \t]*)[A-Za-z_]\w*\s*=\s*verify_all\s*\([^\n]*\)\s*$"
            )
            match = verify_call.search(text)
            if not match:
                raise ValueError(
                    "Could not find the claim-check or citation-verification call "
                    "in pipeline.py. No project files were changed."
                )
            indent = match.group("indent")
            addition = (
                "\n" + indent
                + "# Offline evidence notes enrich uncited-claim RAG only.\n"
                + indent + "load_local_evidence(store)"
            )
            text = text[:match.end()] + addition + text[match.end():]

    return text


DEDUP_HELPER = '''\n\ndef _deduplicate_suggestions(candidates):
    """Keep the highest-ranked suggestion for each normalized paper title."""
    unique = []
    seen = set()
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        title = str(candidate.get("title") or "").strip().casefold()
        key = re.sub(r"[^a-z0-9]+", "", title)
        if not key:
            body = str(candidate.get("text") or "").strip().casefold()
            key = "text:" + re.sub(r"[^a-z0-9]+", "", body[:180])
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
'''


def _set_default_k(text: str) -> str:
    # Support spaces and optional basic type annotations, e.g. k: int = 2.
    pattern = re.compile(
        r"(def\s+check_claims\s*\([^)]*?\bk\s*(?::\s*[^=,]+)?\s*=\s*)\d+",
        re.S,
    )
    return pattern.sub(r"\g<1>3", text, count=1)


def patch_claim_verifier(text: str) -> tuple[str, bool]:
    if not re.search(r"(?m)^import re\s*$", text):
        text = "import re\n" + text

    if "def _deduplicate_suggestions(" not in text:
        marker = "def looks_like_claim(sentence):"
        if marker in text:
            text = text.replace(marker, DEDUP_HELPER + "\n\n" + marker, 1)
        else:
            marker = "def check_claims("
            if marker not in text:
                raise ValueError(
                    "Could not find check_claims() in claim_verifier.py. "
                    "No project files were changed."
                )
            text = text.replace(marker, DEDUP_HELPER + "\n\n" + marker, 1)

    text = _set_default_k(text)

    # Find a one-line assignment from store.search regardless of the destination
    # variable name, quote style, or spaces around the equals sign.
    assignment = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<var>[A-Za-z_]\w*)\s*=\s*"
        r"(?P<rhs>[^\n]*\bstore\.search\([^\n]*\)[^\n]*)$"
    )
    match = assignment.search(text)
    modified_search = False

    if match:
        line = match.group(0)
        # Replace simple k=k / k=2 arguments with over-fetching, while avoiding
        # disrupting expressions that already calculate an over-fetch size.
        if "store.search(" in line and "k=max(k * 4, 12)" not in line:
            updated_line, count = re.subn(
                r"\bk\s*=\s*(?:k|\d+)\b",
                "k=max(k * 4, 12)",
                line,
                count=1,
            )
            if count:
                line = updated_line

        var = match.group("var")
        indent = match.group("indent")
        lines = text.splitlines(keepends=True)
        # Compute the matched line's offset and replace it in place.
        start, end = match.span()
        existing_following = text[end:]
        dedup_statement = f"{indent}{var} = _deduplicate_suggestions({var})[:k]"
        if dedup_statement not in existing_following.splitlines()[:3]:
            replacement = line + "\n" + dedup_statement
            text = text[:start] + replacement + text[end:]
            modified_search = True
        else:
            if line != match.group(0):
                text = text[:start] + line + text[end:]
            modified_search = True
    else:
        # This is non-fatal: the local evidence library can still be loaded into
        # the shared store, and setting default k=3 still helps existing code.
        modified_search = False

    return text, modified_search


def main() -> int:
    missing = [str(p) for p in (PIPELINE, CLAIM_VERIFIER, LOCAL_MODULE, LIBRARY) if not p.exists()]
    if missing:
        print("Missing required files. Extract the ZIP into the CiteGuard project root first:")
        for path in missing:
            print(" -", path)
        return 2

    try:
        pipeline_old = PIPELINE.read_text(encoding="utf-8")
        claim_old = CLAIM_VERIFIER.read_text(encoding="utf-8")
        # Prepare both edits before writing either file.
        pipeline_new = patch_pipeline(pipeline_old)
        claim_new, retrieval_updated = patch_claim_verifier(claim_old)
    except (OSError, ValueError) as exc:
        print(f"Patch not applied: {exc}")
        return 3

    changed = []
    if pipeline_new != pipeline_old:
        _backup(PIPELINE)
        PIPELINE.write_text(pipeline_new, encoding="utf-8")
        changed.append("citeguard/pipeline.py")
    if claim_new != claim_old:
        _backup(CLAIM_VERIFIER)
        CLAIM_VERIFIER.write_text(claim_new, encoding="utf-8")
        changed.append("citeguard/claim_verifier.py")

    print("Local evidence library files are in place.")
    print("Curated notes are loaded after citation verification and before uncited-claim analysis.")
    if retrieval_updated:
        print("Claim retrieval now over-fetches when the existing search call exposes a k argument, deduplicates by title, and returns up to k=3 candidates.")
    else:
        print("NOTE: Could not identify a simple one-line store.search assignment to over-fetch/deduplicate.")
        print("The local library is still integrated; review claim_verifier.py manually if you need custom retrieval handling.")
    if changed:
        print("Updated:", ", ".join(changed))
        print("Backups use the suffix .pre_local_evidence.bak")
    else:
        print("Integration was already applied; no source files needed changing.")
    print("Next: python -m unittest discover -s tests -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
