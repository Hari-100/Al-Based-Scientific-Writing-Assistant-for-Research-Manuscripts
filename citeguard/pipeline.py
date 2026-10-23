from citeguard.parser import parse_manuscript
from citeguard.vector_store import VectorStore
from citeguard.citation_verifier import verify_all
from citeguard.claim_verifier import check_claims
from citeguard.local_evidence import load_local_evidence
from citeguard.aggregator import aggregate


def run_pipeline(text):
    parsed = parse_manuscript(text)
    store = VectorStore()

    # citation checks run first, this populates the store with real abstracts
    citation_results = verify_all(parsed, store)

    # claim checks reuse whatever the citation checks just retrieved
    # Offline evidence notes enrich uncited-claim RAG only.
    load_local_evidence(store)
    claim_results = check_claims(parsed, store)

    return aggregate(citation_results, claim_results)
