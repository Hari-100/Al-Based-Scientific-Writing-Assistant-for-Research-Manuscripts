import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

from citeguard.local_evidence import DEFAULT_LIBRARY_PATH, load_local_evidence


class FakeStore:
    def __init__(self):
        self.records = []

    def add(self, text, metadata):
        self.records.append((text, metadata))


class LocalEvidenceTests(unittest.TestCase):
    def test_library_is_curated_and_well_formed(self):
        payload = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        records = payload["records"]
        self.assertGreaterEqual(len(records), 20)
        ids = [record["id"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))
        for record in records:
            self.assertTrue(record["title"])
            self.assertTrue(record["summary"])
            parsed = urlparse(record["url"])
            self.assertEqual(parsed.scheme, "https")
            self.assertTrue(parsed.netloc)

    def test_loader_adds_every_record_without_network_calls(self):
        payload = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        store = FakeStore()
        loaded = load_local_evidence(store)
        self.assertEqual(loaded, len(payload["records"]))
        self.assertEqual(len(store.records), loaded)
        text, metadata = store.records[0]
        self.assertIn("Evidence summary (paraphrased)", text)
        self.assertEqual(metadata["source"], "local_evidence_library")
        self.assertEqual(metadata["text_kind"], "curated_paraphrase")
        self.assertTrue(metadata["source_url"].startswith("https://"))

    def test_no_duplicate_title_records_in_library(self):
        payload = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        titles = [record["title"].casefold() for record in payload["records"]]
        self.assertEqual(len(titles), len(set(titles)))

    def test_records_cover_both_demo_tracks_and_claim_topics(self):
        payload = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        by_id = {record["id"]: record for record in payload["records"]}
        required = {
            "vaswani_2017_transformer", "devlin_2019_bert",
            "liu_2019_roberta", "beltagy_2019_scibert",
            "lewis_2020_rag", "kingma_2014_adam",
            "goodfellow_2014_gan", "hochreiter_1997_lstm",
            "reimers_2019_sbert", "karpukhin_2020_dpr",
            "thakur_2021_beir", "kusupati_2022_matryoshka",
        }
        self.assertTrue(required.issubset(by_id))


if __name__ == "__main__":
    unittest.main()
