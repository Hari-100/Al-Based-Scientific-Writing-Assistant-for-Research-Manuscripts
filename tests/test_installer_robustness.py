import importlib.util
import unittest
from pathlib import Path

PATCH_SCRIPT = Path(__file__).resolve().parents[1] / "apply_local_evidence_patch.py"
spec = importlib.util.spec_from_file_location("citeguard_patch_installer", PATCH_SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerRobustnessTests(unittest.TestCase):
    def test_handles_original_item_sentence_search_format(self):
        source = '''import re\ndef looks_like_claim(sentence):\n    return True\ndef check_claims(parsed_sentences, store, k=2):\n    suggestions = store.search(item["sentence"], k=k)\n    return suggestions\n'''
        patched, changed = installer.patch_claim_verifier(source)
        compile(patched, "claim_verifier.py", "exec")
        self.assertTrue(changed)
        self.assertIn("def check_claims(parsed_sentences, store, k=3):", patched)
        self.assertIn("k=max(k * 4, 12)", patched)
        self.assertIn("_deduplicate_suggestions(suggestions)[:k]", patched)

    def test_handles_single_quotes_and_different_search_variable(self):
        source = '''import re\ndef looks_like_claim(sentence):\n    return True\ndef check_claims(parsed_sentences, store, k = 2):\n    candidates = store.search(item['sentence'],k = k)\n    return candidates\n'''
        patched, changed = installer.patch_claim_verifier(source)
        compile(patched, "claim_verifier.py", "exec")
        self.assertTrue(changed)
        self.assertIn("candidates = store.search(item['sentence'],k=max(k * 4, 12))", patched)
        self.assertIn("candidates = _deduplicate_suggestions(candidates)[:k]", patched)

    def test_tolerates_search_using_sentence_variable(self):
        source = '''import re\ndef looks_like_claim(sentence):\n    return True\ndef check_claims(parsed_sentences, store, k: int = 2):\n    found = store.search(sentence, k=2)\n    return found\n'''
        patched, changed = installer.patch_claim_verifier(source)
        compile(patched, "claim_verifier.py", "exec")
        self.assertTrue(changed)
        self.assertIn("found = _deduplicate_suggestions(found)[:k]", patched)

    def test_pipeline_loads_library_before_claim_check_and_is_idempotent(self):
        source = '''from citeguard.parser import parse_manuscript\nfrom citeguard.vector_store import VectorStore\nfrom citeguard.citation_verifier import verify_all\nfrom citeguard.claim_verifier import check_claims\nfrom citeguard.aggregator import aggregate\ndef run_pipeline(text):\n    parsed = parse_manuscript(text)\n    store = VectorStore()\n    citation_results = verify_all(parsed, store)\n    claim_results = check_claims(parsed, store)\n    return aggregate(citation_results, claim_results)\n'''
        patched = installer.patch_pipeline(source)
        compile(patched, "pipeline.py", "exec")
        self.assertLess(patched.index("load_local_evidence(store)"), patched.index("claim_results = check_claims"))
        self.assertEqual(installer.patch_pipeline(patched), patched)


if __name__ == "__main__":
    unittest.main()
