import unittest
from unittest.mock import patch

from citeguard.parser import parse_manuscript
from citeguard import citation_apis


class ParserSmokeTests(unittest.TestCase):
    def test_author_year_and_multiple_citations(self):
        parsed = parse_manuscript(
            "BERT uses bidirectional pretraining (Devlin et al., 2019). "
            "Models include GPT and BERT (Radford et al., 2019; Brown et al., 2020)."
        )
        citations = [citation for item in parsed for citation in item["citations"]]
        self.assertEqual(citations, [
            "Devlin et al., 2019", "Radford et al., 2019", "Brown et al., 2020"
        ])

    def test_local_context_distinguishes_models_in_a_list(self):
        parsed = parse_manuscript(
            "The models include GPT (Radford et al., 2019), BERT (Devlin et al., 2019), "
            "and RoBERTa (Liu et al., 2019)."
        )
        records = parsed[0]["citation_contexts"]
        by_citation = {r["citation"]: r["context"] for r in records}
        self.assertIn("GPT", by_citation["Radford et al., 2019"])
        self.assertIn("BERT", by_citation["Devlin et al., 2019"])
        self.assertIn("RoBERTa", by_citation["Liu et al., 2019"])

    def test_reference_section_removed_from_body(self):
        parsed = parse_manuscript(
            "BERT is a language model (Devlin et al., 2019).\n\nReferences\n"
            "Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers."
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["citations"], ["Devlin et al., 2019"])


class ResolverGuardTests(unittest.TestCase):
    def tearDown(self):
        citation_apis.resolve_citation.cache_clear()
        citation_apis._search_openalex.cache_clear()
        citation_apis._search_openaire.cache_clear()
        citation_apis._search_crossref.cache_clear()

    def test_wrong_crossref_record_without_context_is_rejected(self):
        wrong = {
            "title": "Chapter-031 Recognition of Child in Shock",
            "abstract": "",
            "year": 2017,
            "authors": ["Vaswani"],
            "source": "crossref",
            "doi": None,
            "url": None,
            "sources": ["crossref"],
        }
        with patch.object(citation_apis, "_search_openalex", return_value=[]), \
             patch.object(citation_apis, "_search_openaire", return_value=[]), \
             patch.object(citation_apis, "_search_crossref", return_value=[wrong]):
            result = citation_apis.resolve_citation(
                "Vaswani et al., 2017",
                "Transformers use self attention instead of recurrence and convolution, enabling greater parallelization (Vaswani et al., 2017)."
            )
        self.assertIsNone(result)

    def test_relevant_record_is_accepted(self):
        right = {
            "title": "Attention Is All You Need",
            "abstract": "The Transformer uses attention mechanisms instead of recurrence and convolution, enabling greater parallelization.",
            "year": 2017,
            "authors": ["Ashish Vaswani"],
            "source": "openalex",
            "doi": None,
            "url": None,
            "sources": ["openalex"],
        }
        with patch.object(citation_apis, "_search_openalex", return_value=[right]), \
             patch.object(citation_apis, "_search_openaire", return_value=[]), \
             patch.object(citation_apis, "_search_crossref", return_value=[]):
            result = citation_apis.resolve_citation(
                "Vaswani et al., 2017",
                "Transformers use self attention instead of recurrence and convolution, enabling greater parallelization (Vaswani et al., 2017)."
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Attention Is All You Need")


if __name__ == "__main__":
    unittest.main()
