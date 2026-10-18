import unittest
from unittest.mock import patch

from citeguard import citation_apis


class AuthorYearFallbackTests(unittest.TestCase):
    def tearDown(self):
        citation_apis.resolve_citation.cache_clear()
        citation_apis._search_openalex.cache_clear()
        citation_apis._search_openaire.cache_clear()
        citation_apis._search_crossref.cache_clear()

    def test_short_author_year_fallback_finds_vaswani(self):
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
        right = {
            "title": "Attention Is All You Need",
            "abstract": (
                "The Transformer uses attention mechanisms instead of recurrence "
                "and convolution, enabling greater parallelization."
            ),
            "year": 2017,
            "authors": ["Ashish Vaswani"],
            "source": "openalex",
            "doi": None,
            "url": None,
            "sources": ["openalex"],
        }

        def openalex(query):
            return [right] if query == "vaswani 2017" else []

        def openaire(query):
            return []

        def crossref(query, surname):
            return [wrong] if query != "vaswani 2017" else []

        with patch.object(citation_apis, "_search_openalex", side_effect=openalex), \
             patch.object(citation_apis, "_search_openaire", side_effect=openaire), \
             patch.object(citation_apis, "_search_crossref", side_effect=crossref):
            result = citation_apis.resolve_citation(
                "Vaswani et al., 2017",
                "Transformers use self-attention instead of recurrence and convolution, "
                "enabling greater parallelization (Vaswani et al., 2017).",
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Attention Is All You Need")

    def test_existing_reliable_result_does_not_need_fallback(self):
        right = {
            "title": "SciBERT: A Pretrained Language Model for Scientific Text",
            "abstract": "SciBERT is pretrained on scientific publications for scientific NLP tasks.",
            "year": 2019,
            "authors": ["Iz Beltagy"],
            "source": "openalex",
            "doi": None,
            "url": None,
            "sources": ["openalex"],
        }

        with patch.object(citation_apis, "_search_openalex", return_value=[right]) as oa, \
             patch.object(citation_apis, "_search_openaire", return_value=[]) as on, \
             patch.object(citation_apis, "_search_crossref", return_value=[]) as cr:
            result = citation_apis.resolve_citation(
                "Beltagy et al., 2019",
                "SciBERT was pretrained on scientific publications to improve language representations (Beltagy et al., 2019).",
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["title"], right["title"])
        on.assert_not_called()
        cr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
